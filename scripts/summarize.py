"""Print the final test L2RE of the pre-training run and the twelve fine-tuning runs of one size in the
column order of Table 4. Usage: python scripts/summarize.py ti|s|m [runs folder]"""
import json
import sys

COLUMNS = [('ns2d_fno_1e-5', '1e-5'), ('ns2d_fno_1e-4', '1e-4'), ('ns2d_fno_1e-3', '1e-3'),
           ('ns2d_pdb_M1_eta1e-1_zeta1e-1', '(1,0.1)'), ('ns2d_pdb_M1_eta1e-2_zeta1e-2', '(1,0.01)'),
           ('ns2d_pdb_M1e-1_eta1e-1_zeta1e-1', '(0.1,0.1)'), ('ns2d_pdb_M1e-1_eta1e-2_zeta1e-2', '(0.1,0.01)'),
           ('dr_pdb', 'DR'), ('swe_pdb', 'SWE'), ('ns2d_pda', 'NS'), ('ns2d_cond_pda', 'NS-c'), ('cfdbench', 'CFD')]


def read(path, name):
    try:
        with open(path) as f:
            return json.load(f)['test_l2re'][name]
    except (FileNotFoundError, KeyError):
        return None


size, runs = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'runs'
rows = {'pre-trained': [read(f'{runs}/scoformer_{size}_pretrain/results.json', n) for n, _ in COLUMNS],
        'fine-tuned': [read(f'{runs}/scoformer_{size}_finetune_{n}/results.json', n) for n, _ in COLUMNS]}
print(f'{"SCO-" + size.upper():12s}' + ''.join(f'{c:>11s}' for _, c in COLUMNS))
for label, values in rows.items():
    print(f'{label:12s}' + ''.join(f'{v:11.5f}' if v is not None else f'{"-":>11s}' for v in values))
