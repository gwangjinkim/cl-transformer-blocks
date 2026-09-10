"""Create and verify an unsupported native architecture for Python-worker tests."""
import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTConfig, OPTForCausalLM
from transformers import PreTrainedTokenizerFast

torch.set_num_threads(1)

IDS = torch.tensor([[3, 4, 5, 6], [7, 5, 4, 0]])
MASK = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
LABELS = torch.tensor([[3, 4, 5, 6], [7, 5, 4, -100]])


def fixture(destination):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(71)
    config = OPTConfig(vocab_size=32, hidden_size=16, ffn_dim=32, num_hidden_layers=2,
                       num_attention_heads=4, max_position_embeddings=128,
                       dropout=0.0, attention_dropout=0.0, activation_dropout=0.0,
                       word_embed_proj_dim=16, bos_token_id=1, eos_token_id=2,
                       pad_token_id=0)
    OPTForCausalLM(config).float().eval().save_pretrained(root)
    words = ['<pad>', '<bos>', '<eos>', 'Common', 'Lisp', 'runs', 'Python', 'models']
    words += [f'token{i}' for i in range(len(words), 32)]
    tokenizer = Tokenizer(models.WordLevel(dict(zip(words, range(32))), unk_token='<pad>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<pad>', pad_token='<pad>',
                                   bos_token='<bos>', eos_token='<eos>')
    fast.chat_template = "{% for message in messages %}{{ message['role'] }}: {{ message['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"
    fast.save_pretrained(root)
    reference(root)


def reference(root):
    root = Path(root)
    model = AutoModelForCausalLM.from_pretrained(root, local_files_only=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    with torch.no_grad():
        logits = model(input_ids=IDS, attention_mask=MASK).logits
        embeds = model.get_input_embeddings()(IDS)
        embed_logits = model(inputs_embeds=embeds, attention_mask=MASK).logits
        generated = model.generate(IDS[:1, :2], max_new_tokens=4, do_sample=False,
                                   eos_token_id=None, pad_token_id=0)[0]
        beam = model.generate(IDS[:1, :2], max_new_tokens=4, do_sample=False,
                              num_beams=3, num_return_sequences=2, eos_token_id=None,
                              pad_token_id=0, return_dict_in_generate=True,
                              output_scores=True)
        torch.manual_seed(1234)
        sampled = model.generate(IDS[:1, :2], max_new_tokens=4, do_sample=True,
                                 top_k=5, temperature=0.8, eos_token_id=None,
                                 pad_token_id=0)
    trained = AutoModelForCausalLM.from_pretrained(root, local_files_only=True).train()
    optimizer = torch.optim.AdamW(trained.parameters(), lr=0.005, betas=(0.8, 0.9),
                                  eps=1e-6, weight_decay=0.01)
    losses = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = trained(input_ids=IDS, attention_mask=MASK, labels=LABELS).loss
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trained.parameters(), 1.0)
        optimizer.step()
    trained.eval()
    with torch.no_grad():
        trained_logits = trained(input_ids=IDS, attention_mask=MASK).logits
    trained.save_pretrained(root / 'reference-trained')
    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(root, local_files_only=True),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=2, lora_alpha=4,
                   target_modules=['q_proj', 'v_proj'], lora_dropout=0.0))
    with torch.no_grad():
        offset = 1
        for name, parameter in peft_model.named_parameters():
            if 'lora_A.' in name or 'lora_B.' in name:
                values = torch.arange(offset, offset + parameter.numel(), dtype=torch.float32)
                parameter.copy_(values.reshape_as(parameter) / 1000)
                offset += parameter.numel()
    peft_model.eval()
    peft_model.save_pretrained(root / 'python-lora', safe_serialization=True)
    with torch.no_grad():
        peft_logits = peft_model(input_ids=IDS, attention_mask=MASK).logits
    text = 'Common Lisp runs Python models'
    messages = [{'role': 'user', 'content': 'Common Lisp'},
                {'role': 'assistant', 'content': 'Python'}]
    chat_rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    chat_tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True)
    if hasattr(chat_tokens, 'get'):
        chat_tokens = chat_tokens['input_ids']
    (root / 'worker-reference.json').write_text(json.dumps({
        'input_ids': IDS.tolist(), 'attention_mask': MASK.tolist(),
        'shape': list(logits.shape), 'logits': logits.tolist(),
        'labels': LABELS.tolist(), 'inputs_embeds': embeds.tolist(),
        'embed_logits': embed_logits.tolist(), 'adamw_losses': losses,
        'trained_logits': trained_logits.tolist(),
        'peft_logits': peft_logits.tolist(),
        'generated': generated.tolist(), 'text': text,
        'beam_sequences': beam.sequences.tolist(),
        'beam_scores': beam.sequences_scores.tolist(),
        'sampled': sampled.tolist(),
        'encoded': tokenizer.encode(text, add_special_tokens=False),
        'decoded': tokenizer.decode(tokenizer.encode(text, add_special_tokens=False)),
        'chat_rendered': chat_rendered, 'chat_tokens': chat_tokens,
    }))


def verify(source, exported, trained=False):
    source = Path(source)
    source_model = AutoModelForCausalLM.from_pretrained(
        source / 'reference-trained' if trained else source, local_files_only=True).eval()
    exported_model = AutoModelForCausalLM.from_pretrained(exported, local_files_only=True).eval()
    assert source_model.config.model_type == exported_model.config.model_type == 'opt'
    for name, value in source_model.state_dict().items():
        torch.testing.assert_close(value, exported_model.state_dict()[name],
                                   atol=1e-6 if trained else 0,
                                   rtol=1e-4 if trained else 0)
    with torch.no_grad():
        torch.testing.assert_close(source_model(IDS).logits, exported_model(IDS).logits,
                                   atol=2e-6 if trained else 0,
                                   rtol=2e-5 if trained else 0)
    source_tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    exported_tokenizer = AutoTokenizer.from_pretrained(exported, local_files_only=True)
    assert source_tokenizer.encode('Common Lisp runs Python models') == \
        exported_tokenizer.encode('Common Lisp runs Python models')
    print(f'Fresh Python reload matches worker export: {exported}')


def verify_peft(source, adapter, merged):
    source = Path(source)
    adapted = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(source, local_files_only=True),
        adapter, local_files_only=True).eval()
    merged_model = AutoModelForCausalLM.from_pretrained(merged, local_files_only=True).eval()
    with torch.no_grad():
        adapted_logits = adapted(input_ids=IDS, attention_mask=MASK).logits
        merged_logits = merged_model(input_ids=IDS, attention_mask=MASK).logits
    torch.testing.assert_close(adapted_logits, merged_logits, atol=2e-6, rtol=2e-5)
    assert (Path(adapter) / 'adapter_config.json').is_file()
    assert (Path(adapter) / 'adapter_model.safetensors').is_file()
    assert (Path(merged) / 'model.safetensors').is_file()
    print(f'Fresh Python PEFT and merged reload agree: {adapter} -> {merged}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['fixture', 'verify', 'verify-peft'])
    parser.add_argument('path')
    parser.add_argument('exported', nargs='?')
    parser.add_argument('merged', nargs='?')
    parser.add_argument('--trained', action='store_true')
    args = parser.parse_args()
    if args.action == 'fixture':
        fixture(args.path)
    elif args.action == 'verify':
        verify(args.path, args.exported, args.trained)
    else:
        verify_peft(args.path, args.exported, args.merged)
