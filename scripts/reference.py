"""Independent Transformers fixtures and native-export validation (development only)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers
from safetensors.torch import save_file, load_file

torch.set_num_threads(1)

def fixture(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128,
                         rms_norm_eps=1e-5, tie_word_embeddings=True,
                         attention_dropout=0.0, bos_token_id=1, eos_token_id=2,
                         pad_token_id=0)
    model = LlamaForCausalLM(config).float().eval()
    model.set_attn_implementation('eager')
    model.save_pretrained(root)
    words = ['<unk>', '<bos>', '<eos>', 'Common', 'Lisp', 'is', 'fast', 'and', 'fun']
    words += [f'token{i}' for i in range(len(words), 32)]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(32))), unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
                                  bos_token='<bos>', eos_token='<eos>', pad_token='<unk>')
    fast.save_pretrained(root)
    sharded = root / 'sharded'
    model.save_pretrained(sharded, max_shard_size='1KB')
    # Malformed artifact fixtures must fail for the intended schema violation.
    weights = load_file(str(root / 'model.safetensors'))
    untied = root / 'untied'
    untied.mkdir(exist_ok=True)
    untied_config = json.loads((root / 'config.json').read_text())
    untied_config['tie_word_embeddings'] = False
    (untied / 'config.json').write_text(json.dumps(untied_config))
    untied_weights = dict(weights)
    untied_weights['lm_head.weight'] = weights['model.embed_tokens.weight'].clone()
    save_file(untied_weights, str(untied / 'model.safetensors'))
    for asset in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json']:
        if (root / asset).exists(): (untied / asset).write_bytes((root / asset).read_bytes())
    for kind in ['missing', 'extra', 'shape']:
        destination = root / kind
        destination.mkdir(exist_ok=True)
        (destination / 'config.json').write_bytes((root / 'config.json').read_bytes())
        changed = dict(weights)
        if kind == 'missing': del changed['model.norm.weight']
        elif kind == 'extra': changed['surprise.weight'] = torch.ones(1)
        else: changed['model.norm.weight'] = torch.ones(3)
        save_file(changed, str(destination / 'model.safetensors'))
    reference(root)

def reference(root):
    root = Path(root)
    model = AutoModelForCausalLM.from_pretrained(root, dtype=torch.float32,
                                                attn_implementation='eager', local_files_only=True).eval()
    ids = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
    output = model(ids, labels=ids)
    with torch.no_grad():
        greedy = torch.tensor([[3, 4]])
        for _ in range(4):
            greedy = torch.cat((greedy, model(greedy).logits[:, -1].argmax(-1, keepdim=True)), dim=1)
    (root / 'generation-reference.json').write_text(json.dumps(greedy[0].tolist()))
    (root / 'reference.json').write_text(json.dumps({'input_ids': ids.tolist(),
        'logits': output.logits.detach().tolist(), 'loss': output.loss.item()}))
    if model.config.hidden_size == 16:
        output.loss.backward()
        save_file({name: p.grad.contiguous() for name, p in model.named_parameters()},
                  str(root / 'reference-gradients.safetensors'))
        with torch.no_grad():
            for p in model.parameters():
                p.sub_(0.01 * p.grad)
        updated = root / 'reference-updated'
        model.save_pretrained(updated)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    text = 'Common Lisp is fast and fun'
    (root / 'tokenizer-reference.json').write_text(json.dumps({'text': text,
        'ids': tokenizer.encode(text, add_special_tokens=False),
        'decoded': tokenizer.decode(tokenizer.encode(text, add_special_tokens=False))}))
    print(f'Reference ready: {root}')

def verify(source, exported, updated=False):
    src, dst = Path(source), Path(exported)
    a = AutoModelForCausalLM.from_pretrained(src / 'reference-updated' if updated else src,
        dtype=torch.float32, attn_implementation='eager', local_files_only=True).eval()
    b = AutoModelForCausalLM.from_pretrained(dst, dtype=torch.float32,
        attn_implementation='eager', local_files_only=True).eval()
    assert a.config.to_dict()['model_type'] == b.config.to_dict()['model_type']
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=2e-6 if updated else 0,
                                   rtol=2e-5 if updated else 0)
    ids = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 3]])
    with torch.no_grad():
        torch.testing.assert_close(a(ids).logits, b(ids).logits, atol=2e-5, rtol=2e-4)
    ta = AutoTokenizer.from_pretrained(src, local_files_only=True)
    tb = AutoTokenizer.from_pretrained(dst, local_files_only=True)
    assert ta.encode('Common Lisp is fast and fun') == tb.encode('Common Lisp is fast and fun')
    if a.config.tie_word_embeddings:
        assert a.model.embed_tokens.weight is a.lm_head.weight
        assert b.model.embed_tokens.weight is b.lm_head.weight
    print(f'Fresh Transformers reload matches: {dst}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['fixture','reference','verify'])
    parser.add_argument('path')
    parser.add_argument('exported', nargs='?')
    parser.add_argument('--updated', action='store_true')
    args = parser.parse_args()
    if args.action == 'fixture': fixture(args.path)
    elif args.action == 'reference': reference(args.path)
    else: verify(args.path, args.exported, args.updated)
