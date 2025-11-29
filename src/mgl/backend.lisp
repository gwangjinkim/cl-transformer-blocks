(in-package #:cl-transformer-blocks-mgl)

(defparameter *device* :cpu
  "Either :cpu or :gpu.

On Mac (no CUDA) use :cpu. On a CUDA/NVIDIA machine, :gpu can be used
inside WITH-GPU to run operations on the GPU (assuming MGL-MAT CUDA is configured).")

(defmacro with-cpu (() &body body)
  "Evaluate BODY with *DEVICE* bound to :CPU."
  `(let ((*device* :cpu))
     ,@body))

(defmacro with-gpu (() &body body)
  "Evaluate BODY with MGL-MAT CUDA context and *DEVICE* bound to :GPU.

Requires CUDA + NVIDIA GPU. On systems without CUDA, this will signal
an error when MGL-MAT tries to enter the CUDA context."
  `(mgl-mat:with-cuda* ()
     (let ((*device* :gpu))
       ,@body)))
