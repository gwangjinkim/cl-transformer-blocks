# cl-transformer-blocks

A small, extensible **Transformer building block library for Common Lisp**, inspired by [TransformerBlocks.jl](https://github.com/JuliaMLTools/TransformerBlocks.jl) and built on top of **[MGL](https://github.com/melisgl/mgl)** / **[MGL-MAT](https://github.com/melisgl/mgl-mat)**.

- Core: backend-agnostic abstractions for attention, feed-forward layers, and Transformer blocks.
- MGL backend: concrete tensors, layer norm, dropout, and math implemented with `mgl-mat:mat` (CPU or CUDA).

It’s meant as a **hackable playground** for Transformer architecture experiments in Common Lisp, not a fully polished framework.

---

## Status

- ✅ Core API and MGL backend are usable.
- ✅ Simple CPU examples and unit tests.
- ⚠️ CUDA support depends on a working MGL-MAT CUDA setup.
- 🧪 API may still change; expect breaking changes while exploring.

---

## Installation

1. Install **Quicklisp** and **MGL / MGL-MAT** (see their docs).
2. Clone this project into your Quicklisp local projects directory:

   ```sh
   cd ~/quicklisp/local-projects
   git clone https://github.com/your-user/cl-transformer-blocks.git
   ```

3. In your Lisp:

   ```lisp
   (ql:quickload '(:cl-transformer-blocks
                   :cl-transformer-blocks-mgl))
   ```

Packages:

- Core: `cl-transformer-blocks` (nickname: `TB`)
- MGL backend: `cl-transformer-blocks-mgl` (nickname: `TB-MGL`)

You’ll usually `use` or `:nicknames` them:

```lisp
(defpackage #:my-transformer-playground
  (:use #:cl)
  (:local-nicknames (#:tb     #:cl-transformer-blocks)
                    (#:tb-mgl #:cl-transformer-blocks-mgl)))
(in-package #:my-transformer-playground)
```

---

## Quickstart (CPU)

Minimal example: single Transformer block on CPU with random data.

```lisp
(in-package #:my-transformer-playground)

;; Model and sequence sizes
(defparameter *d-model*  8)   ; embedding size
(defparameter *seq-len*  4)   ; sequence length

(tb-mgl:with-cpu ()
  (let* (;; create a random matrix of shape (D_MODEL x SEQ_LEN)
         (x     (tb-mgl:random-mat *d-model* *seq-len*))
         ;; construct one Transformer block on the MGL backend
         (tblock (tb-mgl:make-block *d-model*
                                   :n-heads         2
                                   :ffn-hidden-dim  16
                                   :dropout         0.1d0))
         ;; forward pass (no mask, training mode off => dropout disabled)
         (y     (tb:forward tblock x :training-p nil)))
    (format t "~&Input  shape: ~S~%" (mgl-mat:mat-dimensions x))
    (format t "Output shape: ~S~%" (mgl-mat:mat-dimensions y))))
```

Expected: same shape in, same shape out (one block preserves the feature dimension).

---

## Using CUDA / GPU

The MGL backend is designed to run on GPU if you have CUDA configured for MGL-MAT.

Typical setup in your app:

```lisp
;; Set global defaults at startup (in your own code, not the library):
(setf mgl-mat:*default-mat-ctype* :float)
(setf mgl-mat:*cuda-enabled* t)

;; Then run your computation inside WITH-GPU:
(tb-mgl:with-gpu ()
  (let* ((x     (tb-mgl:random-mat *d-model* *seq-len*))
         (tblock (tb-mgl:make-block *d-model* :n-heads 2 :ffn-hidden-dim 16 :dropout 0.1d0))
         (y     (tb:forward tblock x :training-p t)))
    (format t "GPU output shape: ~S~%" (mgl-mat:mat-dimensions y))))
```

Notes:

- `with-gpu` enters an `mgl-mat:with-cuda*` context and binds `tb-mgl:*device*` to `:gpu`.
- `with-cpu` does the same with `:cpu`, so you can easily switch for debugging.

---

## Stacked Transformer Blocks

You usually want **N Transformer blocks in a row**. Use `make-block-list`:

```lisp
(defun make-encoder-stack (n d-model)
  "Return a list-like encoder with N identical blocks."
  (tb-mgl:make-block-list n d-model
                          :n-heads         4
                          :ffn-hidden-dim  (* 4 d-model)
                          :dropout         0.1d0))

(tb-mgl:with-cpu ()
  (let* ((x      (tb-mgl:random-mat *d-model* *seq-len*))
         (stack  (make-encoder-stack 6 *d-model*))
         ;; training-p T => dropout active
         (y      (tb:forward stack x :training-p t)))
    (format t "Stack output: ~S~%" (mgl-mat:mat-dimensions y)))
```

`make-block-list` returns a `block-list` object that internally holds N `transformer-block` instances and just calls `forward` on them sequentially.

---

## API Overview

### Core package: `cl-transformer-blocks` (`TB`)

The core is backend-agnostic. It knows *what* a Transformer block is, but not *how* to multiply matrices.

Key concepts:

- **Classes**
  - `attention-layer`
  - `feedforward-layer`
  - `transformer-block`
  - `block-list`

- **Constructors**
  - `(tb:make-block backend d-model &key n-heads ffn-hidden-dim dropout)`
  - `(tb:make-block-list backend n d-model &key n-heads ffn-hidden-dim dropout)`

  In practice you usually call the backend-specific wrappers (`tb-mgl:make-block`, `tb-mgl:make-block-list`), which fill in `backend` for you.

- **Forward pass**

  ```lisp
  (tb:forward layer x &key mask training-p)
  ```

  - `layer` can be an `attention-layer`, `transformer-block`, or `block-list`.
  - `x` is the backend tensor (for MGL backend: `mgl-mat:mat`).
  - `mask` is currently assumed to be compatible with the backend format (e.g. a matrix or NIL).
  - `training-p` controls whether dropout is applied.

- **Backend protocol (`tb-` generics)**

  The core defines a set of generics like:

  - `tb-zeros` / `tb-ones`
  - `tb-add`, `tb-matmul`
  - `tb-softmax`
  - `tb-layer-norm`
  - `tb-dropout`
  - (…and a few more primitives used by attention / feed-forward)

  Backends implement these generics for their tensor type and backend keyword.

  Example generic (simplified):

  ```lisp
  (defgeneric tb-zeros (backend dims &key dtype)
    (:documentation "Return a zero-filled tensor for BACKEND with shape DIMS."))
  ```

  The MGL backend implements a method specialized on `:mgl` and `mgl-mat:mat`.

---

### MGL backend: `cl-transformer-blocks-mgl` (`TB-MGL`)

This backend implements the protocol using `mgl-mat:mat` as the tensor type.

Main exports (conceptually):

- Context and utilities:
  - `*device*` – current device (`:cpu` or `:gpu`).
  - `with-cpu`, `with-gpu` – dynamic device selection.
  - `random-mat` – convenience function for creating random matrices for demos.
  - possibly `backend` constant or helper (depending on your final design).

- Constructors:
  - `make-block` – thin wrapper around `tb:make-block` with `backend` fixed to `:mgl`.
  - `make-block-list` – ditto for stacks.

- Tensor operations (methods, not usually called directly):
  - `tb:tb-zeros`, `tb:tb-ones`, `tb:tb-add`, `tb:tb-matmul`, `tb:tb-softmax`,
    `tb:tb-layer-norm`, `tb:tb-dropout`, …

Example: random mat + block, GPU:

```lisp
(tb-mgl:with-gpu ()
  (let* ((x     (tb-mgl:random-mat *d-model* *seq-len*))
         (tblock (tb-mgl:make-block *d-model* :n-heads 2 :ffn-hidden-dim 16 :dropout 0.1d0))
         (y     (tb:forward tblock x :training-p t)))
    (format t "CUDA run ok, shape: ~S~%" (mgl-mat:mat-dimensions y)))
```

---

## LayerNorm & Dropout (MGL backend details)

The MGL backend contains naïve implementations of layer normalization and dropout in terms of `mgl-mat:mref` / `mgl-mat:make-mat` / `mgl-mat:uniform-random!`.

- **LayerNorm**: row-wise mean/variance, then normalize and apply `gamma`/`beta`.
- **Dropout**: build a mask with uniform random numbers, keep a fraction `1-p`, and rescale by `1/(1-p)` in training mode; in evaluation mode, it just copies the input.

You don’t normally call these functions directly; they are invoked from the core `transformer-block` via `tb:tb-layer-norm` and `tb:tb-dropout`.

---

## Running the Tests

The tests use [FiveAM](https://github.com/sionescu/fiveam) and cover shapes and simple statistical properties of the MGL backend.

From the REPL:

```lisp
(ql:quickload :cl-transformer-blocks/tests)
(asdf:test-system :cl-transformer-blocks/tests)
```

You should see something like:

- `ZEROS-SHAPE` – `tb-zeros` produces the right dimensions.
- `RANDOM-MAT-SHAPE` – `random-mat` dimensions.
- `SINGLE-BLOCK-FORWARD-SHAPE` – one block keeps shape stable.
- `STACKED-BLOCKS-FORWARD-SHAPE` – N blocks also keep shape.
- `LAYER-NORM-SHAPE-AND-STATS` – normalization behaves sensibly.
- `DROPOUT-SHAPE-AND-MODE` – dropout toggles with `training-p`.

If something fails, this is a good starting point for debugging your backend changes.

---

## Writing Your Own Backend (Advanced)

The core idea of `cl-transformer-blocks` is that **all numerics go through a tiny protocol**. To support another numeric library or representation:

1. Define your own package, e.g. `cl-transformer-blocks-mybackend`.
2. Pick a backend designator, e.g. `:mybackend`.
3. Implement methods for all required `tb:` generics
   (`tb-zeros`, `tb-add`, `tb-matmul`, `tb-softmax`, `tb-layer-norm`, `tb-dropout`, …)
   specialized on `(:mybackend <your-tensor-type>)`.
4. Optionally provide convenience constructors:

   ```lisp
   (defun make-block (d-model &rest args)
     (apply #'tb:make-block :mybackend d-model args))

   (defun make-block-list (n d-model &rest args)
     (apply #'tb:make-block-list :mybackend n d-model args))
   ```

5. Now the rest of the code (core `transformer-block`, `block-list`, etc.) stays unchanged; it just calls `tb:forward`, which dispatches into your backend.

---

## Design Philosophy

- **Small surface area** – a few generics define the whole “tensor language”.
- **Backend plug-ability** – everything above the `tb-*` level is pure Lisp and backend-agnostic.
- **Hackability over generality** – this is a good codebase to read, modify, and extend if you want to learn how transformers work in detail.

---

## License

Add appropriate license text here (e.g. MIT).

---

## Contributing

- Issues and PRs are welcome.
- If you add a new backend or a more efficient kernel (e.g. fused attention / FlashAttention-style ops), feel free to document it with a small example and a benchmark snippet.
