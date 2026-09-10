"""Unit checks for worker boundaries that must not contact external services."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import unittest
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def load_worker():
    spec = importlib.util.spec_from_file_location('tb_hf_worker', ROOT / 'scripts/hf-worker.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SavedObject:
    def __init__(self, filename):
        self.filename = filename
        self.save_options = []

    def save_pretrained(self, destination, **kwargs):
        self.save_options.append(kwargs)
        Path(destination, self.filename).write_text('fixture', encoding='utf-8')


class HubBoundary(unittest.TestCase):
    def test_save_artifacts_forwards_max_shard_size(self):
        worker = load_worker()
        worker.model = SavedObject('model.safetensors')
        worker.tokenizer = None
        worker.processor = None
        with TemporaryDirectory() as directory:
            worker.save_artifacts(directory, max_shard_size=2048)
        self.assertEqual(
            worker.model.save_options,
            [{'safe_serialization': True, 'max_shard_size': 2048}],
        )

    def test_binary_tensor_rejects_unowned_path_and_wrong_length(self):
        worker = load_worker()
        with TemporaryDirectory() as directory:
            outside = Path(directory, 'tensor.bin')
            outside.write_bytes(b'\0' * 8)
            with self.assertRaisesRegex(ValueError, 'outside'):
                worker.request_value({'kind': 'binary_tensor', 'path': str(outside),
                                      'shape': [1], 'dtype': 'int64'})
        inside = worker.binary_directory / 'bad-length.bin'
        inside.write_bytes(b'\0' * 3)
        try:
            with self.assertRaisesRegex(ValueError, 'byte length'):
                worker.request_value({'kind': 'binary_tensor', 'path': str(inside),
                                      'shape': [1], 'dtype': 'float32'})
        finally:
            inside.unlink(missing_ok=True)

    def test_image_file_descriptor_loads_pixels(self):
        worker = load_worker()
        with TemporaryDirectory() as directory:
            path = Path(directory, 'pixel.png')
            Image.new('RGB', (2, 3), color=(10, 20, 30)).save(path)
            image = worker.processor_value(
                {'kind': 'file', 'media_type': 'image', 'path': str(path)})
            self.assertEqual(image.size, (2, 3))

    def test_hub_publish_stages_complete_artifacts_and_calls_api(self):
        worker = load_worker()
        worker.model = SavedObject('model.safetensors')
        worker.tokenizer = SavedObject('tokenizer.json')
        worker.processor = SavedObject('preprocessor_config.json')
        calls = {}

        class FakeApi:
            def create_repo(self, repo_id, **kwargs):
                calls['create'] = (repo_id, kwargs)

            def upload_folder(self, **kwargs):
                calls['upload'] = kwargs
                folder = Path(kwargs['folder_path'])
                calls['files'] = sorted(path.name for path in folder.iterdir())
                return types.SimpleNamespace(
                    commit_url='https://huggingface.co/acme/model/commit/abc', oid='abc')

        request = {'repo_id': 'acme/model', 'private': True, 'revision': 'dev',
                   'create_pr': True, 'commit_message': 'Publish from Lisp',
                   'commit_description': 'Acceptance fixture', 'model_card': None}
        with mock.patch.object(worker, 'HfApi', FakeApi):
            result = worker.push_to_hub(request)

        self.assertEqual(result['oid'], 'abc')
        self.assertEqual(calls['create'], ('acme/model', {'private': True, 'exist_ok': True}))
        self.assertEqual(calls['upload']['revision'], 'dev')
        self.assertEqual(calls['upload']['commit_message'], 'Publish from Lisp')
        self.assertEqual(calls['files'],
                         ['README.md', 'model.safetensors', 'preprocessor_config.json',
                          'tokenizer.json'])


if __name__ == '__main__':
    unittest.main()
