"""Independent PEFT fixtures, gradients, optimizer trajectories, and export checks."""
import argparse
import copy
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from safetensors.torch import save_file, load_file

torch.set_num_threads(1)
IDS = torch.tensor([[3, 4, 5, 6, 0], [7, 5, 4, 3, 8]])
MASK = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
LABELS = torch.tensor([[-100, -100, 5, 6, -100], [-100, 5, -100, 3, 8]])

def base(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                         attn_implementation='eager', local_files_only=True).eval()

def trainable(model):
    return {k.replace('.default.', '.'): p for k, p in model.named_parameters() if p.requires_grad}

def fixture(base_path, destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    for variant in ['standard', 'rslora']:
        torch.manual_seed(29)
        cfg = LoraConfig(r=2 if variant == 'standard' else 3, lora_alpha=4,
                         target_modules=['q_proj', 'v_proj', 'down_proj'],
                         lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
                         use_rslora=variant == 'rslora')
        model = get_peft_model(base(base_path), cfg).eval()
        model.peft_config['default'].base_model_name_or_path = str(Path(base_path).resolve())
        with torch.no_grad():
            for p in trainable(model).values(): p.uniform_(-0.08, 0.08)
        folder = root / variant
        model.save_pretrained(folder)
        out = model(IDS, attention_mask=MASK, labels=LABELS)
        out.loss.backward()
        (folder / 'reference.json').write_text(json.dumps({'input_ids': IDS.tolist(),
            'attention_mask': MASK.tolist(), 'labels': LABELS.tolist(),
            'logits': out.logits.detach().tolist(), 'loss': out.loss.item()}))
        save_file({k:p.grad.contiguous() for k,p in trainable(model).items()}, str(folder/'gradients.safetensors'))
        for algorithm in ['sgd', 'adamw']:
            updated = PeftModel.from_pretrained(base(base_path), folder, is_trainable=True).eval()
            params = list(trainable(updated).values())
            opt = (torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.02)
                   if algorithm == 'sgd' else torch.optim.AdamW(params, lr=0.001,
                     betas=(0.8, 0.95), eps=1e-8, weight_decay=0.02, foreach=False))
            losses = []
            for _ in range(3):
                opt.zero_grad()
                loss = updated(IDS, attention_mask=MASK, labels=LABELS).loss
                losses.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=0.05, foreach=False)
                opt.step()
            updated.save_pretrained(folder / algorithm)
            (folder / f'{algorithm}-losses.json').write_text(json.dumps(losses))
        print('PEFT reference ready:', folder)
    model = base(base_path)
    out = model(IDS, attention_mask=MASK, labels=LABELS)
    out.loss.backward()
    folder = root / 'full'
    folder.mkdir(exist_ok=True)
    (folder/'reference.json').write_text(json.dumps({'loss': out.loss.item()}))
    save_file({k:p.grad.contiguous() for k,p in model.named_parameters()}, str(folder/'gradients.safetensors'))
    opt=torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.8,0.95), weight_decay=0.02, foreach=False)
    losses=[]
    for _ in range(3):
        opt.zero_grad()
        loss=model(IDS,attention_mask=MASK,labels=LABELS).loss
        losses.append(loss.item());loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),0.05,foreach=False)
        opt.step()
    model.save_pretrained(folder/'adamw')
    (folder/'adamw-losses.json').write_text(json.dumps(losses))

def verify(base_path, source, exported, merged=False):
    expected = PeftModel.from_pretrained(base(base_path), source).eval()
    actual = base(exported) if merged else PeftModel.from_pretrained(base(base_path), exported).eval()
    with torch.no_grad():
        torch.testing.assert_close(actual(IDS, attention_mask=MASK).logits,
                                   expected(IDS, attention_mask=MASK).logits, atol=3e-5, rtol=3e-4)
    if not merged:
        a,b=load_file(str(Path(source)/'adapter_model.safetensors')),load_file(str(Path(exported)/'adapter_model.safetensors'))
        assert a.keys()==b.keys()
        for name in a: torch.testing.assert_close(a[name], b[name], atol=2e-6, rtol=2e-4)
    print('Fresh PEFT/Transformers reload matches:', exported)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['fixture','verify']);p.add_argument('base');p.add_argument('source');p.add_argument('exported',nargs='?');p.add_argument('--merged',action='store_true')
    a=p.parse_args()
    if a.action=='fixture':fixture(a.base,a.source)
    else:verify(a.base,a.source,a.exported,a.merged)
