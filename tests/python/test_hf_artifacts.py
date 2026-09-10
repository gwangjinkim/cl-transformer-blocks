"""No-network tests for the native runtime's Hugging Face artifact helper."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]


def load_helper():
    spec = importlib.util.spec_from_file_location(
        'tb_hf_artifacts', ROOT / 'scripts/hf-artifacts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ArtifactBoundary(unittest.TestCase):
    def test_download_passes_reproducibility_options(self):
        helper = load_helper()
        calls = {}

        def fake_download(repo_id, **kwargs):
            calls.update(repo_id=repo_id, **kwargs)
            return '/cache/models--acme--tiny/snapshots/abc'

        with mock.patch.object(helper, 'snapshot_download', fake_download):
            result = helper.download({
                'repo_id': 'acme/tiny', 'revision': 'abc',
                'local_files_only': True, 'cache_directory': '/cache'})
        self.assertEqual(result['path'], '/cache/models--acme--tiny/snapshots/abc')
        self.assertEqual(calls, {
            'repo_id': 'acme/tiny', 'revision': 'abc',
            'local_files_only': True, 'cache_dir': '/cache',
            'allow_patterns': helper.NATIVE_MODEL_PATTERNS})

    def test_publish_calls_hub_without_receiving_a_token(self):
        helper = load_helper()
        calls = {}

        class FakeApi:
            def create_repo(self, repo_id, **kwargs):
                calls['create'] = (repo_id, kwargs)

            def upload_folder(self, **kwargs):
                calls['upload'] = kwargs
                return types.SimpleNamespace(commit_url='https://example/commit/abc', oid='abc')

        with TemporaryDirectory() as directory, mock.patch.object(helper, 'HfApi', FakeApi):
            Path(directory, 'model.safetensors').write_bytes(b'weights')
            result = helper.publish({
                'repo_id': 'acme/tiny', 'directory': directory, 'private': True,
                'revision': 'dev', 'create_pr': True, 'commit_message': 'Native Lisp'})
        self.assertEqual(result['oid'], 'abc')
        self.assertEqual(calls['create'], ('acme/tiny', {'private': True, 'exist_ok': True}))
        self.assertEqual(Path(calls['upload']['folder_path']), Path(directory).resolve())
        self.assertNotIn('token', calls['upload'])


if __name__ == '__main__':
    unittest.main()
