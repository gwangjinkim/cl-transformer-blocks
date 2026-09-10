"""Native ABI tests; intentionally independent of the Lisp wrappers."""
import ctypes as C
from pathlib import Path
import unittest


class NativeABI(unittest.TestCase):
    def test_matmul_and_ownership(self):
        name = (
            "libtb_mlx.dylib"
            if __import__("sys").platform == "darwin"
            else "libtb_mlx.so"
        )
        lib = C.CDLL(str(Path(".build/native", name).resolve()))
        lib.tb_context_new.argtypes = [C.c_int]
        lib.tb_context_new.restype = C.c_void_p
        lib.tb_tensor_float.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_int), C.c_int]
        lib.tb_tensor_float.restype = C.c_void_p
        lib.tb_matmul.argtypes = [C.c_void_p] * 3
        lib.tb_matmul.restype = C.c_void_p
        lib.tb_tensor_copy_float.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.POINTER(C.c_float),
            C.c_size_t,
        ]
        lib.tb_tensor_free.argtypes = [C.c_void_p]
        lib.tb_context_free.argtypes = [C.c_void_p]
        ctx = lib.tb_context_new(0)
        a = lib.tb_tensor_float((C.c_float * 4)(1, 2, 3, 4), (C.c_int * 2)(2, 2), 2)
        b = lib.tb_matmul(ctx, a, a)
        try:
            self.assertTrue(b)
            output = (C.c_float * 4)()
            self.assertEqual(lib.tb_tensor_copy_float(ctx, b, output, 4), 0)
            self.assertEqual(list(output), [7, 10, 15, 22])
        finally:
            lib.tb_tensor_free(b)
            lib.tb_tensor_free(a)
            lib.tb_context_free(ctx)

    def test_float16_storage_cast_roundtrip(self):
        name = (
            "libtb_mlx.dylib"
            if __import__("sys").platform == "darwin"
            else "libtb_mlx.so"
        )
        lib = C.CDLL(str(Path(".build/native", name).resolve()))
        lib.tb_context_new.argtypes = [C.c_int]
        lib.tb_context_new.restype = C.c_void_p
        lib.tb_tensor_float.argtypes = [C.POINTER(C.c_float), C.POINTER(C.c_int), C.c_int]
        lib.tb_tensor_float.restype = C.c_void_p
        lib.tb_cast_float16.argtypes = [C.c_void_p, C.c_void_p]
        lib.tb_cast_float16.restype = C.c_void_p
        lib.tb_cast_float.argtypes = [C.c_void_p, C.c_void_p]
        lib.tb_cast_float.restype = C.c_void_p
        lib.tb_tensor_dtype.argtypes = [C.c_void_p]
        lib.tb_tensor_dtype.restype = C.c_int
        lib.tb_tensor_copy_float.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.POINTER(C.c_float),
            C.c_size_t,
        ]
        lib.tb_tensor_free.argtypes = [C.c_void_p]
        lib.tb_context_free.argtypes = [C.c_void_p]
        ctx = lib.tb_context_new(0)
        original = lib.tb_tensor_float(
            (C.c_float * 4)(0.1, -0.3333, 2.5, 12.125), (C.c_int * 1)(4), 1
        )
        half = restored = None
        try:
            half = lib.tb_cast_float16(ctx, original)
            self.assertTrue(half)
            self.assertEqual(lib.tb_tensor_dtype(half), 9)
            restored = lib.tb_cast_float(ctx, half)
            self.assertTrue(restored)
            output = (C.c_float * 4)()
            self.assertEqual(lib.tb_tensor_copy_float(ctx, restored, output, 4), 0)
            for actual, expected in zip(output, (0.1, -0.3333, 2.5, 12.125)):
                self.assertAlmostEqual(actual, expected, delta=0.001)
        finally:
            if restored:
                lib.tb_tensor_free(restored)
            if half:
                lib.tb_tensor_free(half)
            lib.tb_tensor_free(original)
            lib.tb_context_free(ctx)

    def test_float16_storage_overflow_is_detectable(self):
        name = (
            "libtb_mlx.dylib"
            if __import__("sys").platform == "darwin"
            else "libtb_mlx.so"
        )
        lib = C.CDLL(str(Path(".build/native", name).resolve()))
        lib.tb_context_new.argtypes = [C.c_int]
        lib.tb_context_new.restype = C.c_void_p
        lib.tb_tensor_float.argtypes = [
            C.POINTER(C.c_float),
            C.POINTER(C.c_int),
            C.c_int,
        ]
        lib.tb_tensor_float.restype = C.c_void_p
        lib.tb_cast_float16.argtypes = [C.c_void_p, C.c_void_p]
        lib.tb_cast_float16.restype = C.c_void_p
        lib.tb_tensor_finite.argtypes = [
            C.c_void_p,
            C.c_void_p,
            C.POINTER(C.c_int),
        ]
        lib.tb_tensor_finite.restype = C.c_int
        lib.tb_tensor_free.argtypes = [C.c_void_p]
        lib.tb_context_free.argtypes = [C.c_void_p]
        ctx = lib.tb_context_new(0)
        original = lib.tb_tensor_float(
            (C.c_float * 1)(70000), (C.c_int * 1)(1), 1
        )
        half = None
        try:
            half = lib.tb_cast_float16(ctx, original)
            self.assertTrue(half)
            finite = C.c_int()
            self.assertEqual(lib.tb_tensor_finite(ctx, half, C.byref(finite)), 0)
            self.assertEqual(finite.value, 0)
        finally:
            if half:
                lib.tb_tensor_free(half)
            lib.tb_tensor_free(original)
            lib.tb_context_free(ctx)

if __name__ == "__main__":
    unittest.main()
