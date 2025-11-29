(in-package #:cl-transformer-blocks)

;;; Abstract numeric protocol – implemented by backends (e.g. MGL-MAT).

(defgeneric tb-matmul (a b)
  (:documentation "Matrix multiplication. Shape semantics are backend-defined."))

(defgeneric tb-add (a b)
  (:documentation "Element-wise addition."))

(defgeneric tb-add-scaled (a b scale)
  (:documentation "Return a + scale * b, fused if backend supports it."))

(defgeneric tb-softmax (x &key axis)
  (:documentation "Softmax of X along AXIS (backend chooses default if NIL)."))

(defgeneric tb-layer-norm (x gamma beta &key epsilon)
  (:documentation "Layer norm over feature dimension.
GAMMA/BETA are optional scale/shift parameters (can be NIL).
EPSILON is a small stabilizer (default backend-specific)."))

(defgeneric tb-dropout (x p &key training-p)
  (:documentation "Dropout with probability P; no-op when TRAINING-P is NIL."))

(defgeneric tb-gelu (x)
  (:documentation "GELU activation function."))

(defgeneric tb-tensor-shape (x)
  (:documentation "Return tensor shape as a list of integers, e.g. (D T)."))

(defgeneric tb-transpose (x)
  (:documentation "Return the transpose of X."))

(defgeneric tb-scale (x alpha)
  (:documentation "Return ALPHA * X."))

(defgeneric tb-zeros (backend dims &key dtype)
  (:documentation "Return a new zero-initialized tensor for BACKEND with shape DIMS.

BACKEND is a designator for the numeric backend (e.g., :MGL).
DIMS is typically a list of dimension sizes, e.g. (D T) or (D T B).
DTYPE is a backend-specific element type or class (e.g., :FLOAT)."))

(defgeneric forward (layer x &key mask training-p)
  (:documentation "Forward pass for LAYER given input tensor X.

MASK and TRAINING-P are optional keyword arguments used by some layers
(e.g. attention masks, dropout behaviour)."))
