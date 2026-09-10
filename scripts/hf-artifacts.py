"""One-shot Hugging Face Hub artifact boundary for native Common Lisp models."""
import json
from pathlib import Path
import sys

from huggingface_hub import HfApi, snapshot_download

NATIVE_MODEL_PATTERNS = [
    '*.json', '*.safetensors', '*.model', '*.txt', '*.jinja', '*.bpe', '*.tiktoken'
]

def download(request):
    options = {
        'revision': request.get('revision'),
        'local_files_only': request.get('local_files_only', False),
        'cache_dir': request.get('cache_directory'),
        'allow_patterns': NATIVE_MODEL_PATTERNS,
    }
    options = {name: value for name, value in options.items() if value is not None}
    path = Path(snapshot_download(request['repo_id'], **options)).resolve()
    return {'path': str(path), 'resolved_revision': path.name}


def publish(request):
    directory = Path(request['directory']).resolve()
    if not directory.is_dir():
        raise ValueError('publication directory does not exist')
    api = HfApi()
    api.create_repo(request['repo_id'], private=request.get('private', False), exist_ok=True)
    commit = api.upload_folder(
        repo_id=request['repo_id'], folder_path=str(directory),
        revision=request.get('revision'), create_pr=request.get('create_pr', False),
        commit_message=request.get('commit_message'),
        commit_description=request.get('commit_description'))
    return {'repo_id': request['repo_id'], 'commit_url': str(commit.commit_url),
            'oid': commit.oid}


def main():
    request = json.load(sys.stdin)
    action = request.get('action')
    if action == 'download':
        result = download(request)
    elif action == 'publish':
        result = publish(request)
    else:
        raise ValueError(f'unknown artifact action: {action}')
    json.dump(result, sys.stdout, ensure_ascii=False, separators=(',', ':'))
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
