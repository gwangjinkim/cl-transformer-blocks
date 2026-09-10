"""Compare C and Lisp with resident FP32 inputs and the same MLX bridge.

Measures synchronized GEMM dispatch/throughput, not full-model generation speed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def command_output(command, **kwargs):
    return subprocess.check_output(list(map(str, command)), cwd=ROOT, text=True, **kwargs).strip()


def measured(command, env):
    output = command_output(command, env=env)
    return json.loads(next(line for line in reversed(output.splitlines()) if line.startswith('{')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output')
    args = parser.parse_args()
    if not 1 <= args.repeats <= 100:
        parser.error('repeats must be 1..100')
    records = []
    for n, iterations in [(16, 400), (128, 200), (512, 40), (1024, 20)]:
        env = {**os.environ, 'TB_DEVICE': args.device, 'TB_BENCH_N': str(n),
               'TB_BENCH_ITERATIONS': str(iterations), 'TB_BENCH_REPEATS': str(args.repeats)}
        results = {}
        # Alternate process order across sizes to reduce a systematic warm-order bias.
        for frontend in (['c', 'lisp'] if n in [16, 512] else ['lisp', 'c']):
            command = ([ROOT / '.build/native/tb_native_benchmark', args.device, n, iterations, args.repeats]
                       if frontend == 'c' else
                       ['sbcl', '--noinform', '--no-userinit', '--no-sysinit', '--script', 'scripts/benchmark.lisp'])
            results[frontend] = measured(command, env)
        if abs(results['c']['verified_value'] - results['lisp']['verified_value']) >= 1e-7:
            raise ValueError('C/Lisp numerical readbacks disagree')
        native = statistics.median(results['c']['seconds'])
        lisp = statistics.median(results['lisp']['seconds'])
        records.append({'dimension': n, 'c_seconds': native, 'lisp_seconds': lisp,
                        'lisp_over_c': lisp / native, 'raw': results})
        print(f'{args.device} {n:4}x{n:<4}: C {native*1e6:9.2f} us, Lisp {lisp*1e6:9.2f} us, ratio {lisp/native:.3f}')
    sources = [*ROOT.glob('src/*.lisp'), *ROOT.glob('native/*.c'), *ROOT.glob('native/*.h'),
               ROOT / 'native/CMakeLists.txt', ROOT / 'scripts/benchmark.lisp',
               ROOT / 'scripts/benchmark.py', ROOT / 'uv.lock', ROOT / 'dependencies/sources.lock.json']
    result = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'machine': platform.machine(),
        'processor': (command_output(['sysctl', '-n', 'machdep.cpu.brand_string'])
                      if platform.system() == 'Darwin' else platform.processor()),
        'device': args.device, 'repeats': args.repeats, 'warmup_iterations': 10,
        'records': records, 'sbcl': command_output(['sbcl', '--version']),
        'compiler': command_output(['cc', '--version']).splitlines()[0],
        'mlx': version('mlx'),
        'cmake_build': [line for line in (ROOT / '.build/native/CMakeCache.txt').read_text().splitlines()
                        if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_C_COMPILER:'))],
        'revision': command_output(['git', 'rev-parse', 'HEAD']),
        'tree_dirty': bool(command_output(['git', 'status', '--porcelain'])),
        'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(sources)},
    }
    output = ROOT / (args.output or f'.build/benchmark-{args.device}.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
