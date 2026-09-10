"""Private JSON-lines worker. Standard output is reserved for protocol frames."""
import atexit
import contextlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory, mkdtemp
import uuid

import numpy as np
import torch
import transformers
from huggingface_hub import HfApi
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, get_scheduler


PROTOCOL_VERSION = 1
TASK_CLASSES = {
    'base': 'AutoModel',
    'causal-lm': 'AutoModelForCausalLM',
    'masked-lm': 'AutoModelForMaskedLM',
    'sequence-classification': 'AutoModelForSequenceClassification',
    'seq2seq-lm': 'AutoModelForSeq2SeqLM',
}
model = None
tokenizer = None
processor = None
actual_device = None
info = None
max_output_elements = 2_000_000
optimizers = {}
optimizer_configs = {}
optimizer_steps = {}
schedulers = {}
scheduler_configs = {}
scalers = {}
training_dtype_name = 'float32'
training_dtype = None
binary_directory = Path(mkdtemp(prefix='cl-transformer-blocks-tensors-')).resolve()
atexit.register(shutil.rmtree, binary_directory, ignore_errors=True)
binary_inputs = 0
binary_outputs = 0


def device_for(requested):
    if requested == 'cpu':
        return 'cpu'
    if requested != 'gpu':
        raise ValueError('device must be cpu or gpu')
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    raise RuntimeError('requested GPU is unavailable; refusing CPU fallback')


def dtype_for(name):
    choices = {'auto': 'auto', 'float32': torch.float32, 'float16': torch.float16,
               'bfloat16': torch.bfloat16}
    if name not in choices:
        raise ValueError(f'unsupported dtype: {name}')
    return choices[name]


def training_dtype_for(name):
    choices = {'float32': None, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    if name not in choices:
        raise ValueError(f'unsupported training dtype: {name}')
    return choices[name]


def model_class_for(task, requested):
    name = requested or TASK_CLASSES.get(task)
    if not isinstance(name, str) or not re.fullmatch(r'AutoModel(?:For[A-Za-z0-9]+)?', name):
        raise ValueError('auto_class must name an installed Transformers AutoModel* class')
    candidate = getattr(transformers, name, None)
    if candidate is None or not hasattr(candidate, 'from_pretrained'):
        raise ValueError(f'unknown Transformers AutoClass: {name}')
    return name, candidate


def load(request):
    global model, tokenizer, processor, actual_device, info, max_output_elements
    global training_dtype_name, training_dtype
    task = request['task']
    auto_class, model_class = model_class_for(task, request.get('auto_class'))
    source = request['source']
    if request.get('num_threads') is not None:
        torch.set_num_threads(int(request['num_threads']))
    kwargs = {'revision': request.get('revision'),
              'local_files_only': request['local_files_only'],
              'trust_remote_code': request['trust_remote_code'],
              'dtype': dtype_for(request['dtype'])}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    actual_device = device_for(request['device'])
    training_dtype_name = request.get('training_dtype', 'float32')
    training_dtype = training_dtype_for(training_dtype_name)
    model = model_class.from_pretrained(source, **kwargs).eval().to(actual_device)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source, revision=request.get('revision'),
            local_files_only=request['local_files_only'],
            trust_remote_code=request['trust_remote_code'])
    except Exception as error:
        tokenizer = None
        tokenizer_error = str(error)
    else:
        tokenizer_error = None
    try:
        processor = AutoProcessor.from_pretrained(
            source, revision=request.get('revision'),
            local_files_only=request['local_files_only'],
            trust_remote_code=request['trust_remote_code'])
    except Exception as error:
        processor = None
        processor_error = str(error)
    else:
        processor_error = None
    max_output_elements = int(request['max_output_elements'])
    if not 1 <= max_output_elements <= 100_000_000:
        raise ValueError('max_output_elements must be 1..100000000')
    parameter = next(model.parameters(), None)
    info = {'protocol_version': PROTOCOL_VERSION, 'pid': os.getpid(), 'task': task,
            'auto_class': auto_class,
            'requested_device': request['device'],
            'actual_device': actual_device, 'model_class': type(model).__name__,
            'model_dtype': (str(parameter.dtype).removeprefix('torch.')
                            if parameter is not None else None),
            'training_dtype': training_dtype_name,
            'tokenizer_class': type(tokenizer).__name__ if tokenizer else None,
            'chat_template': bool(tokenizer and getattr(tokenizer, 'chat_template', None)),
            'tokenizer_error': tokenizer_error,
            'processor_class': type(processor).__name__ if processor else None,
            'processor_error': processor_error, 'config': model.config.to_dict(),
            'binary_directory': str(binary_directory)}
    return info


def require_model():
    if model is None:
        raise RuntimeError('model is not loaded')


def require_tokenizer():
    require_model()
    if tokenizer is None:
        raise RuntimeError(f'tokenizer is unavailable: {info["tokenizer_error"]}')


def apply_chat_template(request):
    require_tokenizer()
    messages = request.get('messages')
    if not isinstance(messages, list) or not messages or any(
            not isinstance(message, dict) for message in messages):
        raise ValueError('messages must be a nonempty object list')
    options = processor_mapping(request.get('options', {}), 'chat template options')
    for reserved in ('tokenize', 'add_generation_prompt', 'continue_final_message'):
        if reserved in options:
            raise ValueError(f'chat template option is reserved: {reserved}')
    tokenize = request.get('tokenize', False)
    result = tokenizer.apply_chat_template(
        messages, tokenize=tokenize,
        add_generation_prompt=request.get('add_generation_prompt', False),
        continue_final_message=request.get('continue_final_message', False), **options)
    if not tokenize:
        if not isinstance(result, str):
            raise TypeError('chat template did not return text')
        return {'text': result}
    if hasattr(result, 'get'):
        result = result['input_ids']
    if torch.is_tensor(result):
        result = result.detach().cpu().tolist()
    if not isinstance(result, list) or any(not isinstance(token, int) for token in result):
        raise TypeError('chat template did not return a flat token ID list')
    return {'input_ids': result}


def require_processor():
    require_model()
    if processor is None:
        raise RuntimeError(f'processor is unavailable: {info["processor_error"]}')


def require_no_optimizers():
    if optimizers:
        raise RuntimeError('dispose resident optimizers before changing or merging adapters')


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_value(item) for item in value]
    if hasattr(value, 'value'):
        return json_value(value.value)
    if isinstance(value, Path):
        return str(value)
    return value


def adapter_info(adapter_name=None):
    trainable = sum(parameter.numel() for parameter in model.parameters()
                    if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if not isinstance(model, PeftModel):
        return {'peft_type': None, 'active_adapters': [],
                'trainable_parameters': trainable, 'total_parameters': total,
                'adapters': {}}
    active = list(model.active_adapters)
    selected = adapter_name or (active[0] if active else next(iter(model.peft_config)))
    if selected not in model.peft_config:
        raise ValueError(f'unknown adapter: {selected}')
    config = json_value(model.peft_config[selected].to_dict())
    return {'adapter_name': selected, 'peft_type': config.get('peft_type'),
            'active_adapters': active, 'trainable_parameters': trainable,
            'total_parameters': total, 'use_rslora': config.get('use_rslora', False),
            'adapters': {name: json_value(item.to_dict())
                         for name, item in model.peft_config.items()}}


def make_lora(request):
    global model
    require_no_optimizers()
    if isinstance(model, PeftModel):
        raise RuntimeError('model already has PEFT adapters')
    config = LoraConfig(
        task_type=request.get('task_type'), r=request['rank'],
        lora_alpha=request['alpha'], target_modules=request.get('target_modules'),
        lora_dropout=request.get('dropout', 0.0), bias=request.get('bias', 'none'),
        use_rslora=request.get('use_rslora', False),
        use_dora=request.get('use_dora', False),
        modules_to_save=request.get('modules_to_save'))
    model = get_peft_model(model, config, adapter_name=request.get('adapter_name', 'default'))
    model.eval().to(actual_device)
    info['model_class'] = type(model).__name__
    return adapter_info(request.get('adapter_name'))


def load_adapter(request):
    global model
    require_no_optimizers()
    source = request['source']
    adapter_name = request.get('adapter_name', 'default')
    kwargs = {'revision': request.get('revision'),
              'local_files_only': request.get('local_files_only', False)}
    kwargs = {key: value for key, value in kwargs.items() if value is not None}
    if isinstance(model, PeftModel):
        if adapter_name in model.peft_config:
            raise ValueError(f'adapter already exists: {adapter_name}')
        model.load_adapter(source, adapter_name=adapter_name,
                           is_trainable=request.get('trainable', False), **kwargs)
        model.set_adapter(adapter_name)
    else:
        model = PeftModel.from_pretrained(
            model, source, adapter_name=adapter_name,
            is_trainable=request.get('trainable', False), **kwargs)
    model.eval().to(actual_device)
    info['model_class'] = type(model).__name__
    return adapter_info(adapter_name)


def save_adapter(request):
    if not isinstance(model, PeftModel):
        raise RuntimeError('model has no PEFT adapter')
    destination = Path(request['destination'])
    destination.mkdir(parents=True, exist_ok=True)
    name = request.get('adapter_name')
    model.save_pretrained(destination, safe_serialization=True,
                          selected_adapters=[name] if name else None)
    return {'destination': str(destination), **adapter_info(name)}


def merge_adapter(request):
    global model
    require_no_optimizers()
    if not isinstance(model, PeftModel):
        raise RuntimeError('model has no PEFT adapter')
    names = request.get('adapter_names')
    model = model.merge_and_unload(safe_merge=request.get('safe_merge', True),
                                   adapter_names=names).eval().to(actual_device)
    info['model_class'] = type(model).__name__
    info['config'] = model.config.to_dict()
    return adapter_info()


def tensor(value):
    return torch.tensor(value, dtype=torch.long, device=actual_device)


def request_value(value):
    global binary_inputs
    if isinstance(value, dict) and value.get('kind') == 'binary_tensor':
        path = Path(value.get('path', '')).resolve()
        if path.parent != binary_directory or not path.is_file():
            raise ValueError('binary tensor path is outside the worker transfer directory')
        dtypes = {'int64': np.dtype('<i8'), 'float32': np.dtype('<f4')}
        dtype = dtypes.get(value.get('dtype'))
        shape = value.get('shape')
        if dtype is None or not isinstance(shape, list) or any(
                not isinstance(x, int) or x < 0 for x in shape):
            raise ValueError('invalid binary tensor dtype or shape')
        expected = int(np.prod(shape, dtype=np.int64))
        if path.stat().st_size != expected * dtype.itemsize:
            raise ValueError('binary tensor byte length mismatch')
        array = np.fromfile(path, dtype=dtype, count=expected).reshape(shape)
        binary_inputs += 1
        return torch.from_numpy(array).to(actual_device)
    if isinstance(value, dict) and value.get('kind') == 'tensor':
        dtypes = {'int64': torch.int64, 'float32': torch.float32, 'bool': torch.bool}
        if value.get('dtype') not in dtypes:
            raise ValueError(f'unsupported input dtype: {value.get("dtype")}')
        shape = value.get('shape')
        if not isinstance(shape, list) or any(not isinstance(x, int) or x < 0 for x in shape):
            raise ValueError('tensor input requires a nonnegative shape')
        expected = 1
        for dimension in shape:
            expected *= dimension
        if expected != len(value.get('data', [])):
            raise ValueError('tensor input shape/data length mismatch')
        return torch.tensor(value['data'], dtype=dtypes[value['dtype']],
                            device=actual_device).reshape(shape)
    return value


def processor_value(value):
    if isinstance(value, dict) and value.get('kind') == 'array':
        shape = value.get('shape')
        data = value.get('data', [])
        if not isinstance(shape, list) or any(not isinstance(x, int) or x < 0 for x in shape):
            raise ValueError('processor array requires a nonnegative shape')
        if int(np.prod(shape, dtype=np.int64)) != len(data):
            raise ValueError('processor array shape/data length mismatch')
        dtype = {'int64': np.int64, 'float32': np.float32}.get(value.get('dtype'))
        if dtype is None:
            raise ValueError(f'unsupported processor array dtype: {value.get("dtype")}')
        return np.asarray(data, dtype=dtype).reshape(shape)
    if isinstance(value, dict) and value.get('kind') == 'file':
        path = value.get('path')
        if not isinstance(path, str) or not Path(path).is_file():
            raise ValueError('processor file does not exist')
        media_type = value.get('media_type')
        if media_type == 'image':
            from transformers.image_utils import load_image
            return load_image(path)
        if media_type == 'audio':
            from transformers.audio_utils import load_audio
            return load_audio(path, sampling_rate=value.get('sampling_rate') or 16000)
        if media_type == 'video':
            from transformers.video_utils import load_video
            options = {name: value.get(name) for name in ('num_frames', 'fps')
                       if value.get(name) is not None}
            return load_video(path, **options)[0]
        raise ValueError(f'unsupported processor media type: {media_type}')
    if isinstance(value, list):
        return [processor_value(item) for item in value]
    return value


def processor_mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be an object')
    return {key: processor_value(item) for key, item in value.items()}


def processed_inputs(request):
    require_processor()
    inputs = processor_mapping(request.get('inputs'), 'processor inputs')
    options = processor_mapping(request.get('options', {}), 'processor options')
    if 'return_tensors' in options and options['return_tensors'] != 'pt':
        raise ValueError('processor return_tensors must be pt')
    options['return_tensors'] = 'pt'
    result = processor(**inputs, **options)
    if not hasattr(result, 'items'):
        raise TypeError('processor did not return a named mapping')
    return dict(result.items())


def processed_model_inputs(request):
    values = processed_inputs(request)
    additions = named_inputs({'inputs': request.get('model_inputs', {})}) \
        if request.get('model_inputs') else {}
    duplicates = set(values).intersection(additions)
    if duplicates:
        raise ValueError(f'duplicate processed/model inputs: {sorted(duplicates)}')
    values.update(additions)
    return {name: value.to(actual_device) if torch.is_tensor(value) else value
            for name, value in values.items()}


def processor_generation_inputs(request):
    adapted = {**request, 'options': request.get('processor_options', {})}
    return processed_model_inputs(adapted)


def named_inputs(request):
    values = request.get('inputs')
    if not isinstance(values, dict) or not values:
        raise ValueError('inputs must be a nonempty object')
    return {name: request_value(value) for name, value in values.items()}


def response_tensor(value):
    if not torch.is_tensor(value):
        raise TypeError('requested output is not a tensor')
    if value.dtype == torch.bool:
        dtype = 'bool'
    elif not value.dtype.is_floating_point:
        dtype = 'int64'
        value = value.to(torch.int64)
    else:
        dtype = 'float32'
        value = value.float()
    value = value.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'dtype': dtype,
            'data': value.reshape(-1).tolist()}


def response_binary_tensor(value):
    global binary_outputs
    if value.dtype == torch.bool:
        dtype, numpy_dtype = 'bool', np.dtype('u1')
    elif not value.dtype.is_floating_point:
        dtype, numpy_dtype = 'int64', np.dtype('<i8')
        value = value.to(torch.int64)
    else:
        dtype, numpy_dtype = 'float32', np.dtype('<f4')
        value = value.float()
    value = value.detach().cpu().contiguous()
    path = binary_directory / f'python-{uuid.uuid4().hex}.bin'
    np.asarray(value.numpy(), dtype=numpy_dtype).tofile(path)
    binary_outputs += 1
    return {'transport': 'binary', 'path': str(path), 'shape': list(value.shape),
            'dtype': dtype}


def selected_outputs(output, names, transport='json'):
    if not isinstance(names, list) or not names or any(not isinstance(x, str) for x in names):
        raise ValueError('outputs must be a nonempty string list')
    values = {}
    total = 0
    for name in names:
        value = getattr(output, name, None)
        if value is None and hasattr(output, 'get'):
            value = output.get(name)
        if value is None:
            raise ValueError(f'model output has no field: {name}')
        if not torch.is_tensor(value):
            raise TypeError(f'model output is not a tensor: {name}')
        total += value.numel()
        if total > max_output_elements:
            raise RuntimeError(f'outputs have {total} elements; limit is {max_output_elements}')
        if transport not in ('json', 'binary'):
            raise ValueError('transport must be json or binary')
        values[name] = (response_binary_tensor(value) if transport == 'binary'
                        else response_tensor(value))
    return values


def optimizer_for(request):
    identifier = str(request['optimizer_id'])
    config = request['optimizer']
    if identifier in optimizers:
        if optimizer_configs[identifier] != config:
            raise ValueError('optimizer configuration changed for an existing optimizer')
        return optimizers[identifier]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError('model has no trainable parameters')
    if config['algorithm'] == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=config['learning_rate'],
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'])
    elif config['algorithm'] == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'],
                                      betas=(config['beta1'], config['beta2']),
                                      eps=config['epsilon'], weight_decay=config['weight_decay'])
    else:
        raise ValueError(f'unsupported optimizer: {config["algorithm"]}')
    optimizers[identifier] = optimizer
    optimizer_configs[identifier] = config
    optimizer_steps[identifier] = 0
    scalers[identifier] = torch.amp.GradScaler(
        actual_device, enabled=actual_device == 'cuda' and training_dtype == torch.float16)
    return optimizer


def optimizer_info(identifier):
    identifier = str(identifier)
    if identifier not in optimizers:
        raise ValueError('optimizer has no resident state')
    return {'step': optimizer_steps[identifier],
            'learning_rate': float(optimizers[identifier].param_groups[0]['lr']),
            'scheduler': scheduler_configs.get(identifier, {}).get('name')}


def install_scheduler(identifier, optimizer, config):
    identifier = str(identifier)
    if identifier in schedulers:
        if scheduler_configs[identifier] != config:
            raise ValueError('scheduler configuration changed for an existing optimizer')
        return schedulers[identifier]
    if optimizer_steps[identifier] != 0:
        raise ValueError('scheduler must be configured before the first optimizer step')
    scheduler = get_scheduler(
        config['name'], optimizer, num_warmup_steps=config.get('warmup_steps'),
        num_training_steps=config.get('total_steps'),
        scheduler_specific_kwargs=config.get('options') or None)
    schedulers[identifier] = scheduler
    scheduler_configs[identifier] = config
    return scheduler


def configure_scheduler(request):
    identifier = str(request['optimizer_id'])
    optimizer = optimizer_for(request)
    install_scheduler(identifier, optimizer, request['scheduler'])
    return optimizer_info(identifier)


def train_microbatches(request, batches):
    if not isinstance(batches, list) or not batches:
        raise ValueError('microbatches must be a nonempty list')
    optimizer = optimizer_for(request)
    identifier = str(request['optimizer_id'])
    scaler = scalers[identifier]
    model.train()
    try:
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for inputs in batches:
            context = (torch.autocast(device_type=actual_device, dtype=training_dtype)
                       if training_dtype is not None else contextlib.nullcontext())
            with context:
                output = model(**inputs)
                loss = getattr(output, 'loss', None)
            if loss is None or loss.numel() != 1:
                raise RuntimeError('model did not return a scalar loss; supply its required labels')
            if not torch.isfinite(loss):
                raise FloatingPointError('training loss is not finite')
            losses.append(loss.detach())
            scaler.scale(loss / len(batches)).backward()
        scaler.unscale_(optimizer)
        parameters = [parameter for parameter in model.parameters()
                      if parameter.requires_grad and parameter.grad is not None]
        if not parameters:
            raise RuntimeError('model produced no gradients')
        if any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise FloatingPointError('gradient is not finite')
        limit = request.get('max_grad_norm')
        if limit is not None:
            if not isinstance(limit, (int, float)) or limit <= 0:
                raise ValueError('max_grad_norm must be positive')
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, limit)
        else:
            gradient_norm = torch.sqrt(sum(parameter.grad.float().square().sum()
                                           for parameter in parameters))
        scaler.step(optimizer)
        scaler.update()
        if identifier in schedulers:
            schedulers[identifier].step()
        if any(not torch.isfinite(parameter).all() for parameter in model.parameters()
               if parameter.requires_grad):
            raise FloatingPointError('updated parameter is not finite')
        optimizer_steps[identifier] += 1
        result = optimizer_info(identifier)
        result.update({'loss': float(torch.stack(losses).mean().cpu()),
                       'gradient_norm': float(gradient_norm.detach().cpu()),
                       'microbatches': len(batches)})
        return result
    finally:
        model.eval()


def train_step(request, inputs=None):
    return train_microbatches(
        request, [inputs if inputs is not None else named_inputs(request)])


@contextlib.contextmanager
def generation_seed(seed):
    if seed is None:
        yield
        return
    if not isinstance(seed, int) or not 0 <= seed < 2 ** 63:
        raise ValueError('seed must be an integer in 0..2^63-1')
    state = rng_state()
    try:
        torch.manual_seed(seed)
        if actual_device == 'cuda':
            torch.cuda.manual_seed_all(seed)
        if actual_device == 'mps' and hasattr(torch.mps, 'manual_seed'):
            torch.mps.manual_seed(seed)
        yield
    finally:
        restore_rng_state(state)


def generate_outputs(request, inputs):
    options = processor_mapping(request.get('options', {}), 'generation options')
    if options.get('return_dict_in_generate') is False:
        raise ValueError('return_dict_in_generate must be true')
    options['return_dict_in_generate'] = True
    with torch.no_grad(), generation_seed(request.get('seed')):
        output = model.generate(**inputs, **options)
    return selected_outputs(output, request.get('outputs'))


def save_artifacts(destination, model_card=None, repo_id=None, max_shard_size=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_options = {'safe_serialization': True}
    if max_shard_size is not None:
        save_options['max_shard_size'] = max_shard_size
    model.save_pretrained(destination, **save_options)
    if tokenizer is not None:
        tokenizer.save_pretrained(destination)
    if processor is not None and processor is not tokenizer:
        processor.save_pretrained(destination)
    if model_card is not None:
        if not isinstance(model_card, str):
            raise ValueError('model_card must be a string')
        (destination / 'README.md').write_text(model_card, encoding='utf-8')
    elif repo_id is not None and not (destination / 'README.md').exists():
        (destination / 'README.md').write_text(
            f'---\nlibrary_name: transformers\n---\n\n# {repo_id.split("/")[-1]}\n\n'
            'Exported from Common Lisp with cl-transformer-blocks.\n', encoding='utf-8')
    return destination


def rng_state():
    state = {'cpu': torch.get_rng_state()}
    if actual_device == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state_all()
    if actual_device == 'mps' and hasattr(torch.mps, 'get_rng_state'):
        state['mps'] = torch.mps.get_rng_state()
    return state


def restore_rng_state(state):
    torch.set_rng_state(state['cpu'])
    if actual_device == 'cuda' and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])
    if actual_device == 'mps' and 'mps' in state and hasattr(torch.mps, 'set_rng_state'):
        torch.mps.set_rng_state(state['mps'])


def save_training_checkpoint(request):
    identifier = str(request['optimizer_id'])
    if identifier not in optimizers:
        raise ValueError('optimizer has no resident state')
    destination = save_artifacts(request['destination'])
    bundle = {'format_version': 1, 'optimizer_config': optimizer_configs[identifier],
              'optimizer_state': optimizers[identifier].state_dict(),
              'step': optimizer_steps[identifier], 'rng_state': rng_state(),
              'scheduler_config': scheduler_configs.get(identifier),
              'scheduler_state': schedulers[identifier].state_dict()
              if identifier in schedulers else None,
              'scaler_state': scalers[identifier].state_dict()}
    torch.save(bundle, destination / 'tb_training_state.pt')
    manifest = {'format_version': 1, 'step': optimizer_steps[identifier],
                'artifact_type': 'adapter' if isinstance(model, PeftModel) else 'model',
                'optimizer': optimizer_configs[identifier],
                'scheduler': scheduler_configs.get(identifier),
                'torch_version': torch.__version__}
    (destination / 'tb_training_state.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return manifest


def restore_training_checkpoint(request):
    source = Path(request['source'])
    state_path = source / 'tb_training_state.pt'
    manifest_path = source / 'tb_training_state.json'
    if not state_path.is_file() or not manifest_path.is_file():
        raise ValueError('training checkpoint state files are missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1:
        raise ValueError('unsupported training checkpoint format')
    bundle = torch.load(state_path, map_location='cpu', weights_only=True)
    if bundle.get('format_version') != 1 or bundle.get('optimizer_config') != request['optimizer']:
        raise ValueError('training checkpoint optimizer configuration mismatch')
    optimizer = optimizer_for(request)
    if bundle.get('scheduler_config') is not None:
        install_scheduler(str(request['optimizer_id']), optimizer, bundle['scheduler_config'])
    optimizer.load_state_dict(bundle['optimizer_state'])
    if bundle.get('scaler_state') is not None:
        scalers[str(request['optimizer_id'])].load_state_dict(bundle['scaler_state'])
    identifier = str(request['optimizer_id'])
    optimizer_steps[identifier] = int(bundle['step'])
    if identifier in schedulers:
        schedulers[identifier].load_state_dict(bundle['scheduler_state'])
    if request.get('restore_rng', True):
        restore_rng_state(bundle['rng_state'])
    return {'step': optimizer_steps[identifier]}


def push_to_hub(request):
    repo_id = request['repo_id']
    dry_run = request.get('dry_run_directory')
    if dry_run:
        destination = save_artifacts(dry_run, request.get('model_card'), repo_id)
        return {'repo_id': repo_id, 'dry_run': True,
                'files': sorted(str(path.relative_to(destination))
                                for path in destination.rglob('*') if path.is_file())}
    with TemporaryDirectory(prefix='cl-transformer-blocks-') as temporary:
        destination = save_artifacts(temporary, request.get('model_card'), repo_id)
        api = HfApi()
        api.create_repo(repo_id, private=request.get('private', False), exist_ok=True)
        commit = api.upload_folder(
            repo_id=repo_id, folder_path=destination,
            revision=request.get('revision'), create_pr=request.get('create_pr', False),
            commit_message=request.get('commit_message'),
            commit_description=request.get('commit_description'))
        return {'repo_id': repo_id, 'dry_run': False,
                'commit_url': str(commit.commit_url), 'oid': commit.oid}


def dispatch(request):
    operation = request.get('op')
    if operation == 'load':
        return load(request)
    require_model()
    if operation == 'info':
        return {**info, 'training': model.training, 'optimizer_count': len(optimizers),
                'binary_inputs': binary_inputs, 'binary_outputs': binary_outputs}
    if operation == 'forward':
        inputs = {'input_ids': tensor(request['input_ids'])}
        if request.get('attention_mask') is not None:
            inputs['attention_mask'] = tensor(request['attention_mask'])
        if request.get('token_type_ids') is not None:
            inputs['token_type_ids'] = tensor(request['token_type_ids'])
        with torch.no_grad():
            output = model(**inputs)
        value = getattr(output, 'logits', None)
        if value is None:
            value = getattr(output, 'last_hidden_state', None)
        if value is None:
            raise RuntimeError('model output has neither logits nor last_hidden_state')
        if value.numel() > max_output_elements:
            raise RuntimeError(f'output has {value.numel()} elements; limit is {max_output_elements}')
        value = value.detach().float().cpu().contiguous()
        return {'shape': list(value.shape), 'data': value.reshape(-1).tolist()}
    if operation == 'python_forward':
        with torch.no_grad():
            output = model(**named_inputs(request))
        return selected_outputs(output, request.get('outputs'), request.get('transport', 'json'))
    if operation == 'process':
        return selected_outputs(processed_inputs(request), request.get('outputs'))
    if operation == 'processor_forward':
        with torch.no_grad():
            output = model(**processed_model_inputs(request))
        return selected_outputs(output, request.get('outputs'))
    if operation == 'python_generate':
        return generate_outputs(request, named_inputs(request))
    if operation == 'processor_generate':
        adapted = {**request, 'options': request.get('generation_options', {})}
        return generate_outputs(adapted, processor_generation_inputs(request))
    if operation == 'train_step':
        return train_step(request)
    if operation == 'processor_train_step':
        return train_step(request, processed_model_inputs(request))
    if operation == 'train_microbatches':
        batches = request.get('microbatches')
        if not isinstance(batches, list):
            raise ValueError('microbatches must be a list')
        return train_microbatches(
            request, [named_inputs({'inputs': batch}) for batch in batches])
    if operation == 'configure_scheduler':
        return configure_scheduler(request)
    if operation == 'optimizer_info':
        return optimizer_info(request['optimizer_id'])
    if operation == 'adapter_info':
        return adapter_info(request.get('adapter_name'))
    if operation == 'make_lora':
        return make_lora(request)
    if operation == 'load_adapter':
        return load_adapter(request)
    if operation == 'save_adapter':
        return save_adapter(request)
    if operation == 'merge_adapter':
        return merge_adapter(request)
    if operation == 'drop_optimizer':
        identifier = str(request['optimizer_id'])
        optimizers.pop(identifier, None)
        optimizer_configs.pop(identifier, None)
        optimizer_steps.pop(identifier, None)
        schedulers.pop(identifier, None)
        scheduler_configs.pop(identifier, None)
        scalers.pop(identifier, None)
        return {'dropped': True}
    if operation == 'encode':
        require_tokenizer()
        return tokenizer.encode(request['text'], add_special_tokens=request['add_special_tokens'])
    if operation == 'apply_chat_template':
        return apply_chat_template(request)
    if operation == 'decode':
        require_tokenizer()
        return tokenizer.decode(request['ids'], skip_special_tokens=request['skip_special_tokens'])
    if operation == 'generate':
        inputs = tensor([request['ids']])
        kwargs = {'max_new_tokens': request['max_new_tokens'], 'do_sample': False,
                  'eos_token_id': request.get('eos_token_id')}
        if model.config.pad_token_id is not None:
            kwargs['pad_token_id'] = model.config.pad_token_id
        with torch.no_grad():
            return model.generate(inputs, **kwargs)[0].tolist()
    if operation == 'save':
        destination = save_artifacts(request['destination'],
                                     max_shard_size=request.get('max_shard_size'))
        return {'destination': str(destination)}
    if operation == 'save_training_checkpoint':
        return save_training_checkpoint(request)
    if operation == 'restore_training_checkpoint':
        return restore_training_checkpoint(request)
    if operation == 'push_to_hub':
        return push_to_hub(request)
    if operation == 'close':
        return {'closed': True}
    raise ValueError(f'unknown operation: {operation}')


def respond(frame):
    sys.__stdout__.write(json.dumps(frame, ensure_ascii=False, separators=(',', ':')) + '\n')
    sys.__stdout__.flush()


def main():
    for line in sys.stdin:
        request = None
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                result = dispatch(request)
            respond({'id': request.get('id'), 'ok': True, 'result': result})
            if request.get('op') == 'close':
                break
        except Exception as error:
            respond({'id': request.get('id') if isinstance(request, dict) else None,
                     'ok': False, 'error_type': type(error).__name__, 'message': str(error)})


if __name__ == '__main__':
    main()
