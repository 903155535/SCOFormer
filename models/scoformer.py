"""SCOFormer: auto-regressive operator transformer with routed grouped spectral mixing.

Data flow (Section 3 and Table A.1 of the paper):
    input window [B, X, Y, T_in, d_u] + (x, y, t) coordinates
    -> patch embedding + positional embedding (per frame)
    -> temporal aggregation over the T_in frames
    -> L HybridBlocks (GFFNO + coordinate attention, then FFN)
       The kernel router mixes K basis kernels in the first block.
    -> projection head -> next frame [B, X, Y, 1, d_u]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class HSwish(nn.Module):
    def forward(self, x):
        return x * (F.relu6(x + 3) / 6)


class CoordinateAttention(nn.Module):
    """Coordinate attention (Hou et al., 2021) with reduction ratio 1."""

    def __init__(self, channels):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        h, w = x.shape[-2:]
        y = torch.cat([self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)], dim=2)   # [B, C, H + W, 1]
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = self.conv_h(y_h).sigmoid()
        a_w = self.conv_w(y_w.permute(0, 1, 3, 2)).sigmoid()
        return x * a_w * a_h


class GFFNO(nn.Module):
    """Grouped factorized Fourier layer.

    A real FFT runs along each spatial axis. The first `modes` modes of each axis are mixed within
    `groups` channel groups by mode-specific complex weights, and the two axial outputs are added.
    With n_kernels = K > 1, the weights are a combination of K basis kernels given by the routing vector.
    """

    def __init__(self, channels, modes, groups, n_kernels=1):
        super().__init__()
        assert channels % groups == 0, 'channels must be divisible by groups'
        self.groups, self.modes, self.n_kernels = groups, modes, n_kernels
        cg = channels // groups
        self.weight_y = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_kernels, groups, cg, cg, modes, 2)))
        self.weight_x = nn.Parameter(nn.init.xavier_normal_(torch.empty(n_kernels, groups, cg, cg, modes, 2)))

    @staticmethod
    def _kernel(route, weight):
        # route [B, K], weight [K, G, Cg, Cg, m, 2] -> complex kernel [B, G, Cg, Cg, m]
        return torch.view_as_complex(torch.einsum('bk,kgpqmr->bgpqmr', route, weight).contiguous())

    def forward(self, x, route=None):
        B, C, M, N = x.shape
        G, m = self.groups, self.modes
        if route is None:
            route = torch.full((B, self.n_kernels), 1.0 / self.n_kernels, device=x.device)
        xg = x.view(B, G, C // G, M, N)

        x_ft = torch.fft.rfft(xg, dim=-1, norm='ortho')
        out_ft = x_ft.new_zeros(B, G, C // G, M, N // 2 + 1)
        out_ft[..., :m] = torch.einsum('bgimy,bgioy->bgomy', x_ft[..., :m], self._kernel(route, self.weight_y))
        out_y = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')

        x_ft = torch.fft.rfft(xg, dim=-2, norm='ortho')
        out_ft = x_ft.new_zeros(B, G, C // G, M // 2 + 1, N)
        out_ft[..., :m, :] = torch.einsum('bgixn,bgiox->bgoxn', x_ft[..., :m, :], self._kernel(route, self.weight_x))
        out_x = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        return (out_x + out_y).reshape(B, C, M, N)


class HybridBlock(nn.Module):
    """GroupNorm -> GFFNO + coordinate attention (added) -> residual -> GroupNorm -> FFN -> residual."""

    def __init__(self, width, modes, groups, n_kernels=1, ffn_ratio=1):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, width)
        self.gffno = GFFNO(width, modes, groups, n_kernels)
        self.coord_attn = CoordinateAttention(width)
        self.norm2 = nn.GroupNorm(8, width)
        hidden = int(width * ffn_ratio)
        self.ffn = nn.Sequential(nn.Conv2d(width, hidden, 1), nn.GELU(), nn.Conv2d(hidden, width, 1))

    def forward(self, x, route=None):
        h = self.norm1(x)
        x = x + (self.gffno(h, route) + self.coord_attn(h))
        return x + self.ffn(self.norm2(x))


class KernelRouter(nn.Module):
    """Global average pooling -> MLP (C -> 128 -> 128 -> K, GELU, no bias). Returns unnormalized routing weights."""

    def __init__(self, width, n_kernels, hidden=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Linear(width, hidden, bias=False), nn.GELU(),
                                 nn.Linear(hidden, hidden, bias=False), nn.GELU(),
                                 nn.Linear(hidden, n_kernels, bias=False))

    def forward(self, x):
        return self.mlp(self.pool(x).flatten(1))


class TemporalAggregation(nn.Module):
    """Sum over the input frames of learned per-frame maps W_s applied to cosine-modulated features."""

    def __init__(self, n_timesteps, width):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_timesteps, width, width) / (n_timesteps * width ** 0.5))
        self.gamma = nn.Parameter(2 ** torch.linspace(-10, 10, width).unsqueeze(0))

    def forward(self, x):                                   # x: [B, h, w, T, C]
        t = torch.linspace(0, 1, x.shape[-2], device=x.device).unsqueeze(-1)
        return torch.einsum('tij,...ti->...j', self.w, x * torch.cos(t @ self.gamma))


class SCOFormer(nn.Module):
    """Maps an input window [B, X, Y, T_in, d_u] to the next frame [B, X, Y, 1, d_u].

    width = C, depth = L, n_kernels = K, groups = G (Table 3); modes = m; ffn_ratio = r.
    """

    def __init__(self, width, depth, n_kernels, groups, modes=9, ffn_ratio=1, img_size=128, patch_size=8,
                 n_channels=4, in_timesteps=10, out_layer_dim=32):
        super().__init__()
        self.n_channels = n_channels
        tokens = img_size // patch_size
        embed_hidden = n_channels * patch_size
        self.patch_embed = nn.Sequential(
            nn.Conv2d(n_channels + 3, embed_hidden, kernel_size=patch_size, stride=patch_size), nn.GELU(),
            nn.Conv2d(embed_hidden, width, kernel_size=1))
        self.pos_embed = nn.Parameter(nn.init.trunc_normal_(torch.zeros(1, width, tokens, tokens), std=0.02))
        self.time_agg = TemporalAggregation(in_timesteps, width)
        self.router = KernelRouter(width, n_kernels)
        self.blocks = nn.ModuleList(
            [HybridBlock(width, modes, groups, n_kernels if i == 0 else 1, ffn_ratio) for i in range(depth)])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(width, out_layer_dim, kernel_size=patch_size, stride=patch_size), nn.GELU(),
            nn.Conv2d(out_layer_dim, out_layer_dim, kernel_size=1), nn.GELU(),
            nn.Conv2d(out_layer_dim, n_channels, kernel_size=1))

    @staticmethod
    def coordinates(x):                                     # [B, X, Y, T, 3] with x, y, t in [0, 1]
        B, X, Y, T = x.shape[:4]
        gx = torch.linspace(0, 1, X, device=x.device).view(1, X, 1, 1, 1).repeat(B, 1, Y, T, 1)
        gy = torch.linspace(0, 1, Y, device=x.device).view(1, 1, Y, 1, 1).repeat(B, X, 1, T, 1)
        gt = torch.linspace(0, 1, T, device=x.device).view(1, 1, 1, T, 1).repeat(B, X, Y, 1, 1)
        return torch.cat((gx, gy, gt), dim=-1)

    def forward(self, x):
        B, T = x.shape[0], x.shape[3]
        x = torch.cat((x, self.coordinates(x)), dim=-1).contiguous()
        x = self.patch_embed(rearrange(x, 'b x y t c -> (b t) c x y')) + self.pos_embed
        x = self.time_agg(rearrange(x, '(b t) c x y -> b x y t c', b=B, t=T))
        x = rearrange(x, 'b x y c -> b c x y')

        route = self.router(x)
        for i, block in enumerate(self.blocks):
            x = block(x, route if i == 0 else None)

        x = self.head(x).permute(0, 2, 3, 1)
        return x.reshape(*x.shape[:3], 1, self.n_channels).contiguous()
