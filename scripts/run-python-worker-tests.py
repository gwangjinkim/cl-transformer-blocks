"""Reproduce the explicit Python compatibility-worker acceptance suite."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / '.build/fixtures/python-worker-opt'


def run(*command):
    subprocess.run(list(map(str, command)), cwd=ROOT, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    args = parser.parse_args()
    export = ROOT / f'.build/python-worker-export-{args.device}'
    run(sys.executable, '-m', 'unittest', 'tests/python/test_hf_worker.py')
    run(sys.executable, 'scripts/python-worker-reference.py', 'fixture', FIXTURE)
    subprocess.run(['sbcl', '--noinform', '--no-userinit', '--no-sysinit',
                    '--script', 'scripts/test-python-worker.lisp'], cwd=ROOT, check=True,
                   env={**os.environ, 'TB_DEVICE': args.device})
    run(sys.executable, 'scripts/python-worker-reference.py', 'verify', FIXTURE, export)
    run(sys.executable, 'scripts/python-worker-reference.py', 'verify', FIXTURE,
        ROOT / f'.build/python-worker-sharded-{args.device}')
    run(sys.executable, 'scripts/python-worker-reference.py', 'verify', FIXTURE,
        ROOT / f'.build/python-worker-trained-{args.device}', '--trained')
    for kind in ('loaded', 'trained'):
        run(sys.executable, 'scripts/python-worker-reference.py', 'verify-peft', FIXTURE,
            ROOT / f'.build/python-worker-{kind}-adapter-{args.device}',
            ROOT / f'.build/python-worker-{kind}-merged-{args.device}')
    print(f'All explicit Python worker gates passed: {args.device}.')
