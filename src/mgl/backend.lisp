in-package #:cl-transformer-blocks-mgl)

(defun setup-default-backend (&key (ctype :float) (cuda t))
  "Configure global MGL-MAT defaults for cl-transformer-blocks.

- CTYPE: :float, :double, etc.
- CUDA:  T enables CUDA support (if available), NIL disables it."
  (setf mgl-mat:*default-mat-ctype* ctype
        mgl-mat:*cuda-enabled*      (and cuda t)))

(defmacro with-cpu (() &body body)
  "Run BODY on CPU: disable CUDA and use the current default CTYPE."
  `(let ((*device* :cpu)
         (mgl-mat:*cuda-enabled* nil))
     ,@body))

(defmacro with-gpu (() &body body)
  "Run BODY on GPU: enable CUDA and enter CUDA context.

Requires CUDA + NVIDIA GPU and a properly configured MGL-MAT CUDA."
  `(let ((*device* :gpu)
         (mgl-mat:*cuda-enabled* t))
     (mgl-mat:with-cuda* ()
       ,@body)))
