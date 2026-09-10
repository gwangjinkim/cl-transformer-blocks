"""Optional numerical calibration for the real checkpoint (needs about 2 GB RAM).
Run the real acceptance suite on CPU and GPU first. Compares native logits, Torch
FP32, and Torch FP64 using the same checkpoint and positions, never altered weights.
"""
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM

torch.set_num_threads(1)
root = Path('.build/models/smollm2')
reference = json.loads((root / 'reference.json').read_text())
ids = torch.tensor(reference['input_ids'])
model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float64,
            attn_implementation='eager', local_files_only=True).eval()
with torch.no_grad(): exact = model(ids).logits
candidates = {'torch-fp32': torch.tensor(reference['logits'], dtype=torch.float64)}
for device in ['cpu', 'gpu']:
    path = Path(f'.build/real-logits-{device}.safetensors')
    if path.exists(): candidates[f'mlx-{device}'] = load_file(path)['logits'].double()
for name, value in candidates.items():
    error = (value - exact).abs()
    print(f'{name}: max={error.max():.8g}, rms={error.square().mean().sqrt():.8g}, '
          f'greedy_match={torch.equal(value.argmax(-1), exact.argmax(-1))}')
