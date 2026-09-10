"""Reproduce GPT-2 architecture acceptance against Transformers."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
MODEL_ID='distilbert/distilgpt2'
MODEL_REVISION='2290a62682d06624634c1f46a6ad5be0f47f38aa'
def run(*args,**kwargs):subprocess.run(list(map(str,args)),cwd=ROOT,check=True,**kwargs)
def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','gpu'],default='cpu');p.add_argument('--real',action='store_true');a=p.parse_args()
    os.chdir(ROOT)
    if a.real:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_dir='.build/models/distilgpt2',
                          allow_patterns=['*.json','*.safetensors','*.txt'])
        Path('.build/models/distilgpt2/revision.txt').write_text(MODEL_REVISION+'\n')
        run(sys.executable,'scripts/gpt2-reference.py','reference','.build/models/distilgpt2')
    else:
        run(sys.executable,'scripts/reference.py','fixture','.build/fixtures/tiny')
        run(sys.executable,'scripts/gpt2-reference.py','fixture','.build/fixtures/gpt2')
    environment={**os.environ,'TB_DEVICE':a.device}
    if a.real:environment['TB_GPT2_REAL']='1'
    else:environment.pop('TB_GPT2_REAL',None)
    run('sbcl','--noinform','--no-userinit','--no-sysinit','--script','scripts/test-gpt2.lisp',env=environment)
    for variant in (['distilgpt2'] if a.real else ['standard','untied','scaling','gelu','legacy']):
        folder='.build/models/distilgpt2' if a.real else f'.build/fixtures/gpt2/{variant}'
        run(sys.executable,'scripts/gpt2-reference.py','verify',folder,f'.build/gpt2-export-{variant}-{a.device}')
        if not a.real:
            run(sys.executable,'scripts/gpt2-reference.py','verify',folder,f'.build/gpt2-updated-{variant}-{a.device}','--updated')
    print(f'All GPT-2 acceptance gates passed: {a.device}, real={a.real}')
if __name__=='__main__':main()
