import torch
import yaml


def rel_l2(pred, target, mask=None):
    """Relative L2 error (L2RE) of each sample, averaged over the channels selected by `mask`.

    pred, target: [B, X, Y, T, C]; mask: [B, X, Y, 1, C] with ones at the evaluated points and channels.
    Each channel norm runs over all grid points and frames. Returns a tensor of shape [B].
    """
    B, C = pred.shape[0], pred.shape[-1]
    if mask is not None:
        pred, target = pred * mask, target * mask
        n_channels = mask.sum(dim=list(range(1, mask.ndim - 1))).count_nonzero(dim=-1)
    else:
        n_channels = C
    diff = torch.norm(pred.reshape(B, -1, C) - target.reshape(B, -1, C), 2, dim=1)
    ref = torch.norm(target.reshape(B, -1, C), 2, dim=1) + 1e-8
    return torch.sum(diff / ref, dim=-1) / n_channels


@torch.no_grad()
def rollout_error(model, loader, device=None, gather=None):
    """Auto-regressive rollout over each test trajectory; returns the mean L2RE over trajectories."""
    model.eval()
    errors = []
    for x, y, mask in loader:
        if device is not None:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
        preds = []
        for _ in range(y.shape[-2]):
            out = model(x)
            preds.append(out)
            x = torch.cat((x[..., 1:, :], out), dim=-2)
        err = rel_l2(torch.cat(preds, dim=-2), y, mask)
        errors.append(gather(err) if gather is not None else err)
    return torch.cat(errors).mean().item()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
