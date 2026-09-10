"""Reproduce the native acceptance gates. No cloud resources are provisioned."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'
MODEL_REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'

def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--real', action='store_true', help='Download and verify pinned SmolLM2 (about 270 MB)')
    args = parser.parse_args()
    os.chdir(ROOT)
    fixture = '.build/models/smollm2' if args.real else '.build/fixtures/tiny'
    prefix = 'real-' if args.real else ''
    if args.real:
        from huggingface_hub import snapshot_download
        snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_dir=fixture,
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja'])
        (ROOT / fixture / 'revision.txt').write_text(MODEL_REVISION + '\n')
    run(sys.executable, 'scripts/reference.py', 'reference' if args.real else 'fixture', fixture)
    run(sys.executable, '-m', 'unittest', 'discover', '-s', 'tests/python')
    run('sbcl', '--noinform', '--no-sysinit', '--no-userinit', '--script', 'scripts/test.lisp',
        env={**os.environ, 'TB_DEVICE': args.device, 'TB_FIXTURE': fixture + '/', 'TB_OUTPUT_PREFIX': prefix})
    run(sys.executable, 'scripts/reference.py', 'verify', fixture, f'.build/{prefix}export-{args.device}')
    run(sys.executable, 'scripts/reference.py', 'verify', fixture,
        f'.build/{prefix}sharded-export-{args.device}')
    if not args.real:
        run(sys.executable, 'scripts/reference.py', 'verify', fixture + '/untied',
            f'.build/untied-export-{args.device}')
        run(sys.executable, 'scripts/reference.py', 'verify', fixture, f'.build/updated-{args.device}', '--updated')
    environment = {'platform': platform.platform(), 'machine': platform.machine(),
                   'python': sys.version, 'device': args.device, 'real_checkpoint': args.real,
                   'model_revision': MODEL_REVISION if args.real else None,
                   'packages': {p: importlib.metadata.version(p) for p in
                                ['mlx', 'torch', 'transformers', 'safetensors', 'tokenizers']}}
    for tool in ['sbcl', 'cmake', 'cargo', 'rustc']:
        environment[tool] = subprocess.check_output([tool, '--version'], text=True).splitlines()[0]
    (ROOT / f'.build/validation-{prefix}{args.device}.json').write_text(json.dumps(environment, indent=2) + '\n')
    print(f'All acceptance gates passed: {args.device}, {fixture}')

if __name__ == '__main__':
    main()
