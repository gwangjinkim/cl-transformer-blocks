"""Fetch checksum-pinned sources and build native libraries. Requires Python 3.12."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]

def run(*args, **kwargs):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True, **kwargs)

def main():
    dependencies = ROOT / '.build/deps'
    downloads = ROOT / '.build/downloads'
    dependencies.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)
    sources = json.loads((ROOT / 'dependencies/sources.lock.json').read_text())
    for source in sources:
        archive = downloads / (source['name'] + '.tar.gz')
        if not archive.exists():
            with urllib.request.urlopen(source['url']) as response:
                archive.write_bytes(response.read())
        if hashlib.sha256(archive.read_bytes()).hexdigest() != source['sha256']:
            raise RuntimeError(f"Checksum mismatch: {archive}; remove this cached file and retry")
        # Re-extract pinned sources so local dependency edits cannot silently affect builds.
        with tarfile.open(archive) as tar:
            tar.extractall(dependencies, filter='data')
    mlx_c = next(dependencies / s['directory'] for s in sources if s['name'] == 'mlx-c')
    spec = importlib.util.find_spec('mlx')  # Locate the native SDK without initializing a GPU.
    if spec is None:
        raise RuntimeError('Install locked Python build dependencies with uv sync first')
    mlx = Path(next(iter(spec.submodule_search_locations)))
    run('cmake', '-S', 'native', '-B', '.build/native',
        f'-DMLX_C_SOURCE={mlx_c}', f'-DCMAKE_PREFIX_PATH={mlx}', '-DCMAKE_BUILD_TYPE=Release')
    run('cmake', '--build', '.build/native', '--parallel', os.environ.get('TB_BUILD_JOBS', '4'))
    run('cargo', 'build', '--locked', '--release', '--manifest-path', 'native/tokenizer/Cargo.toml',
        env={**os.environ, 'CARGO_TARGET_DIR': str(ROOT / '.build/tokenizer')})
    print('Native libraries and pinned Lisp dependencies are ready.')

if __name__ == '__main__':
    main()
