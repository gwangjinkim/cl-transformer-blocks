"""Independent GPT-2 fixtures and native-export verification."""
import argparse
import json
from pathlib import Path
import shutil
from collections import Counter
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file, load_file

torch.set_num_threads(1)
IDS=torch.tensor([[3,4,5,6],[7,5,4,3]])
MASK=torch.tensor([[1,1,1,0],[1,1,1,1]])
LABELS=torch.tensor([[-100,4,5,-100],[-100,-100,4,3]])

def load(root):
    return AutoModelForCausalLM.from_pretrained(root,dtype=torch.float32,attn_implementation='eager',local_files_only=True).eval()

def reference(root, training=True):
    root=Path(root);model=load(root)
    out=model(IDS,attention_mask=MASK,labels=LABELS)
    with torch.no_grad():
        ids=torch.tensor([[3,4]])
        for _ in range(4):ids=torch.cat([ids,model(ids).logits[:,-1].argmax(-1,keepdim=True)],dim=1)
    (root/'reference.json').write_text(json.dumps({'input_ids':IDS.tolist(),'attention_mask':MASK.tolist(),
         'labels':LABELS.tolist(),'logits':out.logits.detach().tolist(),'loss':out.loss.item(),'generated':ids[0].tolist()}))
    if training:
        out.loss.backward()
        save_file({name:p.grad.contiguous() for name,p in model.named_parameters()},str(root/'gradients.safetensors'))
        with torch.no_grad():
            for p in model.parameters():p.sub_(0.01*p.grad)
        model.save_pretrained(root/'updated')
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True)
    text='Common Lisp is fast and fun'
    (root/'tokenizer-reference.json').write_text(json.dumps({'text':text,
       'ids':tokenizer.encode(text,add_special_tokens=False),'decoded':tokenizer.decode(tokenizer.encode(text,add_special_tokens=False))}))
    print('GPT-2 reference ready:',root)

def fixture(root):
    root=Path(root)
    for variant in ['standard','untied','scaling','gelu','legacy']:
        torch.manual_seed(41)
        cfg=GPT2Config(vocab_size=32,n_positions=64,n_embd=16,n_layer=2,n_head=4,n_inner=40,
              resid_pdrop=0.0,embd_pdrop=0.0,attn_pdrop=0.0,bos_token_id=1,eos_token_id=2,pad_token_id=0,
              tie_word_embeddings=variant!='untied',scale_attn_weights=variant!='scaling',
              scale_attn_by_inverse_layer_idx=variant=='scaling',activation_function='gelu' if variant=='gelu' else 'gelu_new')
        model=GPT2LMHeadModel(cfg).eval();folder=root/variant
        model.save_pretrained(folder, max_shard_size='3KB' if variant=='scaling' else '5GB')
        for asset in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json']:
            source=Path('.build/fixtures/tiny')/asset
            if source.exists():shutil.copyfile(source,folder/asset)
        if variant=='legacy':
            weights=load_file(str(folder/'model.safetensors'))
            for i in range(cfg.n_layer):
                weights[f'transformer.h.{i}.attn.bias']=torch.tril(torch.ones(cfg.n_positions,cfg.n_positions)).view(1,1,cfg.n_positions,cfg.n_positions)
            save_file(weights,str(folder/'model.safetensors'))
            invalid=root/'invalid-buffer';invalid.mkdir(exist_ok=True)
            shutil.copyfile(folder/'config.json',invalid/'config.json')
            bad={k:v.clone() for k,v in weights.items()}
            bad['transformer.h.0.attn.bias'][0,0,0,0]=0
            save_file(bad,str(invalid/'model.safetensors'))
        reference(folder)
    bad_index=root/'bad-index'
    shutil.copytree(root/'scaling',bad_index,dirs_exist_ok=True)
    index=json.loads((bad_index/'model.safetensors.index.json').read_text())
    counts=Counter(index['weight_map'].values())
    groups=[name for name,count in counts.items() if count>1]
    assert len(groups)>=2
    keys=[next(key for key,value in index['weight_map'].items() if value==name) for name in groups[:2]]
    index['weight_map'][keys[0]],index['weight_map'][keys[1]]=index['weight_map'][keys[1]],index['weight_map'][keys[0]]
    (bad_index/'model.safetensors.index.json').write_text(json.dumps(index))

def verify(source,destination,updated=False):
    source=Path(source);a=load(source/'updated' if updated else source);b=load(destination)
    assert a.state_dict().keys()==b.state_dict().keys()
    for name,p in a.state_dict().items():torch.testing.assert_close(p,b.state_dict()[name],atol=2e-6 if updated else 0,rtol=2e-4 if updated else 0)
    with torch.no_grad():torch.testing.assert_close(a(IDS,attention_mask=MASK).logits,b(IDS,attention_mask=MASK).logits,atol=3e-5,rtol=3e-4)
    assert (b.transformer.wte.weight is b.lm_head.weight)==b.config.tie_word_embeddings
    ta=AutoTokenizer.from_pretrained(source,local_files_only=True);tb=AutoTokenizer.from_pretrained(destination,local_files_only=True)
    assert ta.encode('Common Lisp is fast and fun')==tb.encode('Common Lisp is fast and fun')
    print('Fresh GPT-2 reload matches:',destination)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['fixture','reference','verify']);p.add_argument('path');p.add_argument('exported',nargs='?');p.add_argument('--updated',action='store_true');a=p.parse_args()
    if a.action=='fixture':fixture(a.path)
    elif a.action=='reference':reference(a.path,False)
    else:verify(a.path,a.exported,a.updated)
