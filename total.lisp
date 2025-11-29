;;; FILE: cl-transformer-blocks-core.asd

(asdf:defsystem "cl-transformer-blocks-core"
  :description "Core transformer blocks independent of numeric backend."
  :author "Gwang-Jin Kim <gwang.jin.kim.phd@gmail.com>"
  :license "MIT"
  :serial t
  :pathname "src/core/"
  :components ((:file "package")
               (:file "protocol")
               (:file "attention")
               (:file "feedforward")
               (:file "block")
               (:file "block-list")))


;;; FILE: cl-transformer-blocks-mgl.asd

(asdf:defsystem "cl-transformer-blocks-mgl"
  :description "MGL-MAT backend for cl-transformer-blocks-core (CPU/GPU capable)."
  :author "Gwang-Jin Kim <your@email>"
  :license "MIT"
  :depends-on ("cl-transformer-blocks-core" "mgl-mat")
  :serial t
  :pathname "src/mgl/"
  :components ((:file "package")
               (:file "backend")
               (:file "ops")
               (:file "factory")))


;;; FILE: cl-transformer-blocks.asd

(asdf:defsystem "cl-transformer-blocks"
  :description "TransformerBlocks-style core + MGL backend for Common Lisp"
  :author "Gwang-Jin Kim <gwang.jin.kim.phd@gmail.com>"
  :license "MIT"
  :depends-on ("mgl-mat")
  :components
  ((:module "src"
    :components
    ((:module "core"
      :components
      ((:file "package")
       (:file "protocol")
       (:file "block")))
     (:module "mgl"
      :components
      ((:file "package")
       (:file "backend")
       (:file "factory")
       (:file "ops")
       (:file "layers")))))))

(asdf:defsystem "cl-transformer-blocks/tests"
  :description "Test suite for cl-transformer-blocks"
  :depends-on ("cl-transformer-blocks" "fiveam")
  :components
  ((:module "tests"
    :components
    ((:file "package")
     (:file "mgl-tests" :depends-on ("package")))))
  :perform (asdf:test-op (op system)
             (declare (ignore op system))
             (uiop:symbol-call :cl-transformer-blocks.tests '#:run-tests)))


;;; FILE: examples/mgl-minimal.lisp

(let* ((d          32)
       (time-steps 10)
       (x          (tb-mgl:random-mat d time-steps))
       (blk        (tb-mgl:make-block d))
       (y          (cl-transformer-blocks:forward blk x)))
  (format t "~&in shape:  ~a~%" (tb:tb-tensor-shape x))
  (format t "out shape: ~a~%" (tb:tb-tensor-shape y)))


;;; FILE: src/core/attention.lisp

(in-package #:cl-transformer-blocks)

(defclass attention-layer ()
  ((w-q :initarg :w-q :reader attention-w-q)
   (w-k :initarg :w-k :reader attention-w-k)
   (w-v :initarg :w-v :reader attention-w-v)
   (w-o :initarg :w-o :reader attention-w-o)
   (model-dim :initarg :model-dim :reader attention-model-dim))
  (:documentation
   "Single-head self-attention layer.

Assumes inputs X have shape (D x T), where D = MODEL-DIM."))

(defmethod forward ((layer attention-layer) x &key mask training-p)
  (declare (ignore mask training-p))
  (let* ((d (attention-model-dim layer))
         (w-q (attention-w-q layer))
         (w-k (attention-w-k layer))
         (w-v (attention-w-v layer))
         (w-o (attention-w-o layer))

         ;; Projections: (D x D) * (D x T) -> (D x T)
         (q (tb-matmul w-q x))
         (k (tb-matmul w-k x))
         (v (tb-matmul w-v x))

         ;; Scores: (T x D) * (D x T) -> (T x T)
         (k-t (tb-transpose k))
         (scores (tb-matmul k-t q))

         ;; Scale by 1/sqrt(D)
         (scaled (tb-scale scores (/ 1.0d0 (sqrt (coerce d 'double-float)))))

         ;; Softmax over last axis (backend decides exact axis semantics)
         (weights (tb-softmax scaled :axis -1))

         ;; Attention output: (D x T) = (D x T) * (T x T)
         (attn (tb-matmul v weights))

         ;; Output projection: (D x D) * (D x T) -> (D x T)
         (out (tb-matmul w-o attn)))
    out))


;;; FILE: src/core/block-list.lisp

;;; Composes multiple blocks sequentially.

(in-package #:cl-transformer-blocks)

(defclass block-list ()
  ((blocks :initarg :blocks :reader block-list-blocks))
  (:documentation "Sequential list of modules with a FORWARD method (e.g., BLOCKs)."))

(defun make-block-list (blocks)
  "Wrap a list or vector of module instances into a BLOCK-LIST."
  (make-instance 'block-list :blocks (coerce blocks 'vector)))

(defmethod forward ((bl block-list) x &key mask training-p)
  (let ((result x))
    (loop for blk across (block-list-blocks bl) do
         (setf result (forward blk result :mask mask :training-p training-p)))
    result))


;;; FILE: src/core/block.lisp

;;; Core block abstractions for cl-transformer-blocks.
;;; Backend-agnostic definitions of attention / feed-forward / transformer
;;; blocks and block stacks. All numeric work is delegated to protocol
;;; generics such as FORWARD, TB-LAYER-NORM, TB-ADD, etc.

(in-package #:cl-transformer-blocks)

;;; ------------------------------------------------------------
;;; Concrete-but-generic layer types
;;; ------------------------------------------------------------

(defclass attention-layer ()
  ((w-q
     :initarg :w-q
     :accessor attention-w-q
     :documentation "Projection matrix for queries (shape [d_model, d_model]).")
   (w-k
     :initarg :w-k
     :accessor attention-w-k
     :documentation "Projection matrix for keys (shape [d_model, d_model]).")
   (w-v
     :initarg :w-v
     :accessor attention-w-v
     :documentation "Projection matrix for values (shape [d_model, d_model]).")
   (w-o
     :initarg :w-o
     :accessor attention-w-o
     :documentation "Output projection matrix (shape [d_model, d_model]).")
   (model-dim
     :initarg :model-dim
     :reader attention-model-dim
     :documentation "Model dimension d_model."))
  (:documentation
   "Generic attention layer storing its projection weights.

Backends (MGL, NumCL, etc.) can use these slots to hold their tensor
representation (e.g. MGL-MAT:MAT), and implement FORWARD methods for
(ATTENTION-LAYER <backend-tensor>)."))

(defclass feedforward-layer ()
  ((w1
     :initarg :w1
     :accessor ffn-w1
     :documentation "First weight matrix (shape [hidden_dim, d_model]).")
   (w2
     :initarg :w2
     :accessor ffn-w2
     :documentation "Second weight matrix (shape [d_model, hidden_dim]).")
   (model-dim
     :initarg :model-dim
     :reader ffn-model-dim
     :documentation "Model dimension d_model.")
   (hidden-dim
     :initarg :hidden-dim
     :reader ffn-hidden-dim
     :documentation "Hidden dimension in the FFN."))
  (:documentation
   "Generic position-wise feed-forward layer storing its weight matrices.

The numeric implementation is backend-specific and provided via FORWARD
methods on (FEEDFORWARD-LAYER <backend-tensor>)."))

;;; ------------------------------------------------------------
;;; Transformer block (pre-norm, with residuals)
;;; ------------------------------------------------------------

(defclass transformer-block ()
  ((attention
     :initarg :attention
     :reader block-attention
     :documentation "Self-attention sublayer instance (ATTENTION-LAYER subclass).")
   (ffn
     :initarg :ffn
     :reader block-ffn
     :documentation "Feed-forward sublayer instance (FEEDFORWARD-LAYER subclass).")
   (use-layer-norm
     :initarg :use-layer-norm
     :initform t
     :accessor block-use-layer-norm
     :documentation
     "If true (default), use a simple pre-norm architecture:

        y1 = attention( LN(x) )
        r1 = x + y1

        y2 = ffn( LN(r1) )
        r2 = r1 + y2

      If NIL, layer normalization is skipped and the block becomes:

        y1 = attention(x)
        r1 = x + y1

        y2 = ffn(r1)
        r2 = r1 + y2

      Layer normalization is delegated to the backend via TB-LAYER-NORM."))
  (:documentation
   "A single transformer block: self-attention + feed-forward + residuals.

The actual numeric implementation is backend-specific and provided via
FORWARD, TB-LAYER-NORM and TB-ADD methods on backend tensor types."))

;;; ------------------------------------------------------------
;;; Block stacks
;;; ------------------------------------------------------------

(defclass block-list ()
  ((blocks
     :initarg :blocks
     :accessor block-list-blocks
     :documentation "Sequence (list) of TRANSFORMER-BLOCK instances to apply."))
  (:documentation
   "A stack / list of transformer blocks applied sequentially.

BACKENDS do not need to subclass this; FORWARD is implemented in terms
of the generic FORWARD and TB-ADD on the underlying tensor type."))

(defun make-block-list (blocks)
  "Wrap a list of blocks into a BLOCK-LIST instance.

BLOCKS is typically a list of TRANSFORMER-BLOCK instances created by
a backend helper (e.g. cl-transformer-blocks-mgl:MAKE-BLOCK)."
  (make-instance 'block-list :blocks blocks))

;;; ------------------------------------------------------------
;;; Forward methods 
;;; ------------------------------------------------------------
;;; These methods are backend-agnostic. They assume that:
;;;   - BACKEND subclasses ATTENTION-LAYER and FEEDFORWARD-LAYER and
;;;     defines FORWARD for (their-layer-type tensor).
;;;   - TB-LAYER-NORM and TB-ADD are defined for the backend's tensor
;;;     representation.
        
;;; ------------------------------------------------------------
;;; Block-level FORWARD methods (backend-agnostic)
;;; ------------------------------------------------------------

(defmethod forward ((blk transformer-block) x)
  "Forward pass through a Transformer block.

All numeric work is delegated to backend implementations of FORWARD
(for the attention/ffn layers) and TB-LAYER-NORM/TB-ADD."
  (labels ((maybe-ln (tensor)
             (if (block-use-layer-norm blk)
                 ;; GAMMA/BETA = NIL: backend may use default parameters
                 (tb-layer-norm tensor nil nil)
                 tensor)))
    (let* (;; pre-norm before attention
           (x-ln   (maybe-ln x))
           (attn-y (forward (block-attention blk) x-ln))
           (r1     (tb-add x attn-y))
           ;; pre-norm before feed-forward
           (r1-ln  (maybe-ln r1))
           (ffn-y  (forward (block-ffn blk) r1-ln))
           (r2     (tb-add r1 ffn-y)))
      r2)))

(defmethod forward ((stack block-list) x)
  "Forward pass through a list of blocks, sequentially."
  (reduce #'forward (block-list-blocks stack)
          :initial-value x))


;;; FILE: src/core/feedforward.lisp

(in-package #:cl-transformer-blocks)

(defclass feedforward-layer ()
  ((w1 :initarg :w1 :reader ffn-w1)
   (w2 :initarg :w2 :reader ffn-w2)
   (model-dim  :initarg :model-dim  :reader ffn-model-dim)
   (hidden-dim :initarg :hidden-dim :reader ffn-hidden-dim))
  (:documentation
   "Position-wise feedforward network.

W1: (hidden-dim x model-dim)
W2: (model-dim x hidden-dim)
Input and output shapes: (model-dim x T)."))

(defmethod forward ((layer feedforward-layer) x &key mask training-p)
  (declare (ignore mask))
  (let* ((w1 (ffn-w1 layer))
         (w2 (ffn-w2 layer))
         (hidden (tb-matmul w1 x))    ; (H x T)
         (activated (tb-gelu hidden)) ; (H x T)
         (out (tb-matmul w2 activated))) ; (D x T)
    ;; Dropout is backend’s decision; here we just call protocol
    (tb-dropout out 0.0 :training-p training-p)))


;;; FILE: src/core/package.lisp

(defpackage #:cl-transformer-blocks
  (:use #:cl)
  (:nicknames #:tb)
  (:export
   ;; Protocol generics
   #:tb-matmul
   #:tb-add
   #:tb-add-scaled
   #:tb-softmax
   #:tb-layer-norm
   #:tb-dropout
   #:tb-gelu
   #:tb-tensor-shape
   #:tb-transpose
   #:tb-scale
   #:tb-zeros
   #:forward
   
   ;; Core API
   #:transformer-block
   #:block-list
   #:make-block-list

   ;; Layer classes (for backends to construct)
   #:attention-layer
   #:feedforward-layer

   ;; accessors we need from MGL backend
   #:attention-w-q
   #:attention-w-k
   #:attention-w-v
   #:attention-w-o
   #:attention-model-dim

   #:ffn-w1
   #:ffn-w2
   #:ffn-model-dim
   #:ffn-hidden-dim

   ;; block readers
   #:block-attention
   #:block-ffn
   #:block-use-layer-norm))


;;; FILE: src/core/protocol.lisp

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

(defgeneric tb-layer-norm (x gamma beta &key eps)
  (:documentation "Layer norm over feature dimension."))

(defgeneric tb-dropout (x p &key training-p)
  (:documentation "Dropout with probability P; usually a no-op when TRAINING-P is NIL."))

(defgeneric tb-gelu (x)
  (:documentation "GELU activation function."))

(defgeneric tb-tensor-shape (x)
  (:documentation "Return tensor shape as a list of integers, e.g. (C T B)."))

(defgeneric tb-transpose (x)
  (:documentation "Return the transpose of X."))

(defgeneric tb-scale (x alpha)
  (:documentation "Return alpha * X."))

(defgeneric tb-zeros (backend dims &key dtype)
  (:documentation "Return a new zero-initialized tensor for BACKEND with shape DIMS.

BACKEND is a designator for the numeric backend (e.g., :MGL).
DIMS is typically a list of dimension sizes, e.g. (D T) or (D T B).
DTYPE is a backend-specific element type or class (e.g., :FLOAT)."))

;;; Generic forward for all modules (Block, Block-List, etc.)

(defgeneric forward (layer x &key mask training-p)
  (:documentation "Forward pass for LAYER given input tensor X."))


;;; FILE: src/mgl/backend.lisp

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


;;; FILE: src/mgl/factory.lisp

;;; Backend-specific constructors that actually create wheight matrices and wire up block + sublayers.

;;; src/mgl/block.lisp

(in-package #:cl-transformer-blocks-mgl)

;;; ------------------------------------------------------------
;;; FFN configuration
;;; ------------------------------------------------------------

(defparameter *default-ffn-multiplier* 4
  "Multiplier for the FFN hidden dimension: H = *DEFAULT-FFN-MULTIPLIER* * D.")

;;; ------------------------------------------------------------
;;; Backend-specific constructors that actually create weight
;;; matrices and wire up block + sublayers.
;;; ------------------------------------------------------------

(defun make-attention-layer (model-dim)
  "Create an ATTENTION-LAYER with random initialized weights for MODEL-DIM."
  (make-instance 'cl-transformer-blocks:attention-layer
                 :w-q (random-mat model-dim model-dim)
                 :w-k (random-mat model-dim model-dim)
                 :w-v (random-mat model-dim model-dim)
                 :w-o (random-mat model-dim model-dim)
                 :model-dim model-dim))

(defun make-feedforward-layer (model-dim &key (multiplier *default-ffn-multiplier*))
  "Create a FEEDFORWARD-LAYER with hidden dimension = MULTIPLIER * MODEL-DIM."
  (let ((hidden-dim (* multiplier model-dim)))
    (make-instance 'cl-transformer-blocks:feedforward-layer
                   :w1 (random-mat hidden-dim model-dim)
                   :w2 (random-mat model-dim hidden-dim)
                   :model-dim model-dim
                   :hidden-dim hidden-dim)))

(defun make-block (model-dim &key (ffn-multiplier *default-ffn-multiplier*)
                              (use-layer-norm t))
  "Create a full TRANSFORMER-BLOCK (attention + FFN) with random parameters."
  (let* ((attn (make-attention-layer model-dim))
         (ffn  (make-feedforward-layer model-dim :multiplier ffn-multiplier)))
    (make-instance 'cl-transformer-blocks:transformer-block
                   :attention attn
                   :ffn       ffn
                   :use-layer-norm use-layer-norm)))


;;; FILE: src/mgl/layers.lisp

(in-package #:cl-transformer-blocks-mgl)

;;; We assume:
;;;  - cl-transformer-blocks:attention-layer has accessors:
;;;      ATTENTION-W-Q, ATTENTION-W-K, ATTENTION-W-V, ATTENTION-W-O
;;;  - cl-transformer-blocks:feedforward-layer has accessors:
;;;      FFN-W1, FFN-W2
;;;  - the MGL backend implements TB-MATMUL, TB-TRANSPOSE,
;;;      TB-SOFTMAX, TB-GELU for MGL-MAT:MAT tensors.

;;; ------------------------------------------------------------
;;; Attention-layer forward for MGL backend
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:forward
    ((layer cl-transformer-blocks:attention-layer)
     (x mgl-mat:mat))
  "Backend implementation of self-attention for MGL-MAT tensors."
  (let* ((w-q (cl-transformer-blocks:attention-w-q layer))
         (w-k (cl-transformer-blocks:attention-w-k layer))
         (w-v (cl-transformer-blocks:attention-w-v layer))
         (w-o (cl-transformer-blocks:attention-w-o layer))
         ;; Q, K, V projections
         (q   (cl-transformer-blocks:tb-matmul w-q x))
         (k   (cl-transformer-blocks:tb-matmul w-k x))
         (v   (cl-transformer-blocks:tb-matmul w-v x))
         ;; attention scores (very naive single-head version)
         (scores  (cl-transformer-blocks:tb-matmul
                   (cl-transformer-blocks:tb-transpose k)
                   q))
         (weights (cl-transformer-blocks:tb-softmax scores))
         (ctx     (cl-transformer-blocks:tb-matmul v weights)))
    (cl-transformer-blocks:tb-matmul w-o ctx)))

;;; ------------------------------------------------------------
;;; Feedforward-layer forward for MGL backend
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:forward
    ((layer cl-transformer-blocks:feedforward-layer)
     (x mgl-mat:mat))
  "Backend implementation of position-wise feed-forward layer for MGL-MAT."
  (let* ((w1 (cl-transformer-blocks:ffn-w1 layer))
         (w2 (cl-transformer-blocks:ffn-w2 layer))
         (h  (cl-transformer-blocks:tb-gelu
              (cl-transformer-blocks:tb-matmul w1 x))))
    (cl-transformer-blocks:tb-matmul w2 h)))


;;; FILE: src/mgl/ops.lisp

(in-package #:cl-transformer-blocks-mgl)

;;; --------------------
;;; protocol: allocation
;;; --------------------

(defmethod tb-zeros ((backend (eql :mgl)) dims &key (dtype :float))
  "Create a zero-initialized MGL-MAT:MAT of shape DIMS.
  
BACKEND is ignored except for dispatch; it should be :MGL.
DTYPE is passed to MGL-MAT:MAKE-MAT as CTYPE when it makes sense."
  (declare (ignore backend))
  ;; DIMS is e.g. (rows cols) or (d t b)
  (mgl-mat:make-mat dims :ctype dtype :initial-element 0.0d0))


;;; --------------------
;;; basic allocation
;;; --------------------


(defun random-mat (rows cols &key (stddev 0.02d0))
  "Create a ROWS x COLS MGL-MAT:MAT with small random values in [-SCALE, SCALE]."
  (let ((m (tb:tb-zeros :mgl (list rows cols))))
    (mgl-mat:gaussian-random! m :mean 0 :stddev stddev)
    m))

;; if you really want a uniform in [-sacle, scale], then do:

;; (defun random-mat (rows cols &key (scale 0.02d0))
;;   "Create a ROWS x COLS MGL-MAT:MAT with small random values in [-SCALE, SCALE]."
;;   (let ((m (tb:tb-zeros :mgl (list rows cols))))
;;     (mgl-mat:uniform-random! m :limit (* 2.0d0 scale))
;;     ;; Shift to [-scale, scale).
;;     (mgl-mat:.+! (- scale) m)
;;     m))



;;; --------------------
;;; protocol: basic ops
;;; --------------------

;;; Implement the protocol generics for MGL-MAT:MAT

(defmethod tb-tensor-shape ((x mat))
  (mat-dimensions x))

(defmethod tb-matmul ((a mat) (b mat))
  "Matrix multiplication via GEMM (BLAS/cuBLAS)."
  (mm* a b))

(defmethod tb-add ((a mat) (b mat))
  "Return a + b as a new MAT."
  (let ((y (copy-mat a)))
    (axpy! 1.0d0 b y)  ; y := y + 1.0 * b
    y))

(defmethod tb-add-scaled ((a mat) (b mat) scale)
  "Return a + scale * b as a new MAT."
  (let ((y (copy-mat a)))
    (axpy! scale b y)
    y))

(defmethod tb-scale ((x mat) alpha)
  "Return alpha * x as a new MAT."
  (let ((y (copy-mat x)))
    (scal! alpha y)
    y))

(defmethod tb-transpose ((x mat))
  (transpose x))


;; ;;; --------------------
;; ;;; protocol: GELU
;; ;;; --------------------

;; (defmethod tb-gelu ((x mat))
;;   "Elementwise approximate GELU on MAT."
;;   (let* ((dims (mat-dimensions x))
;;          (rows (first dims))
;;          (cols (second dims))
;;          (y    (copy-mat x)))
;;     (dotimes (i rows)
;;       (dotimes (j cols)
;;         (let* ((v (mref y i j))
;;                (u (/ v (sqrt 2.0d0)))
;;                (g (* 0.5d0 v (+ 1.0d0 (erf u)))))
;;           (setf (mref y i j) g))))
;;     y))

;;; ------------------------------------------------------------
;;; GELU (approximate, no ERF needed)
;;; ------------------------------------------------------------

(defun %gelu-approx (x)
  "Approximate GELU using tanh-based formula (no ERF)."
  ;; Hendrycks & Gimpel approximation:
  ;; GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
  (let* ((c (/ (sqrt (* 2.0d0 pi))))           ; √(2/π)
         (x3 (* x x x))
         (inner (+ x (* 0.044715d0 x3)))
         (tanh-arg (* c inner)))
    (* 0.5d0 x (+ 1.0d0 (tanh tanh-arg)))))

(defmethod tb-gelu ((x mat))
  "Elementwise approximate GELU on MAT."
  (let* ((dims (mat-dimensions x))
         (rows (first dims))
         (cols (second dims))
         (y    (copy-mat x)))
    (dotimes (i rows)
      (dotimes (j cols)
        (let ((v (mref y i j)))
          (setf (mref y i j) (%gelu-approx v)))))
    y))

;;; --------------------
;;; protocol: softmax (2D, axis=-1)
;;; --------------------

(defun %softmax-row! (m row)
  "In-place softmax of ROW of M (2D MAT) using numerical stabilization."
  (let* ((dims (mat-dimensions m))
         (cols (second dims)))
    ;; 1. max for numerical stability
    (let ((max-val -1d300))
      (dotimes (j cols)
        (let ((v (mref m row j)))
          (when (> v max-val)
            (setf max-val v))))
      ;; 2. exponentiate shifted values & accumulate sum
      (let ((sum 0d0))
        (dotimes (j cols)
          (let* ((v  (mref m row j))
                 (ev (exp (- v max-val))))
            (setf (mref m row j) ev)
            (incf sum ev)))
        ;; 3. normalize
        (dotimes (j cols)
          (setf (mref m row j)
                (/ (mref m row j) sum)))))))

(defmethod tb-softmax ((x mat) &key (axis -1))
  "Softmax over the last axis for 2D MATS. Returns a new MAT."
  (unless (eql axis -1)
    (error "tb-softmax (mgl): only AXIS = -1 is supported right now, got ~S"
           axis))
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-softmax (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (y    (copy-mat x)))
      (dotimes (i rows)
        (%softmax-row! y i))
      y)))


;;; --------------------
;;; protocol: layer-norm
;;; --------------------

;;; This implementation only uses mat-dimensions, mref, copy-mat, make-mat (indirectly),
;;; and has no mysterious MGL internals,
;;; and it's correct enough for now. (later optimization with custom kernels possible).


(defun %layer-norm-row! (x out row gamma beta eps)
  "Compute layer-norm for one ROW of X into OUT.

X, OUT are MGL-MAT:MATS of shape (rows x cols).
GAMMA, BETA are either NIL or MATS of shape (1 x cols).
EPS is a small float for numerical stability."
  (let* ((dims (mat-dimensions x))
         (cols (second dims)))
    ;; 1. compute mean and variance for this row
    (let ((sum 0d0)
          (sumsq 0d0))
      (dotimes (j cols)
        (let ((v (mref x row j)))
          (incf sum v)
          (incf sumsq (* v v))))
      (let* ((cols-d (coerce cols 'double-float))
             (mean   (/ sum cols-d))
             (var    (max eps (- (/ sumsq cols-d) (* mean mean))))
             (inv-std (/ 1.0d0 (sqrt var))))
        ;; 2. normalize and apply gamma/beta
        (dotimes (j cols)
          (let* ((v (mref x row j))
                 (norm (* (- v mean) inv-std))
                 (g (if gamma
                        (mref gamma 0 j)
                        1.0d0))
                 (b (if beta
                        (mref beta 0 j)
                        0.0d0))
                 (y (+ (* norm g) b)))
            (setf (mref out row j) y)))))))

(defmethod tb-layer-norm ((x mat) gamma beta &key (eps 1e-5))
  "Layer normalization over the last dimension (features) for 2D MATS.

X is (rows x cols). GAMMA and BETA are either NIL or (1 x cols) MATS
(broadcast across rows)."
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-layer-norm (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (cols (second dims))
           ;; sanity check gamma/beta if provided
           (gamma* (when gamma
                     (let ((gdims (mat-dimensions gamma)))
                       (unless (and (= (length gdims) 2)
                                    (= (first gdims) 1)
                                    (= (second gdims) cols))
                         (error "tb-layer-norm (mgl): GAMMA must be (1 x ~D), got ~S"
                                cols gdims))
                       gamma)))
           (beta*  (when beta
                     (let ((bdims (mat-dimensions beta)))
                       (unless (and (= (length bdims) 2)
                                    (= (first bdims) 1)
                                    (= (second bdims) cols))
                         (error "tb-layer-norm (mgl): BETA must be (1 x ~D), got ~S"
                                cols bdims))
                       beta)))
           (out (copy-mat x)))
      (dotimes (i rows)
        (%layer-norm-row! x out i gamma* beta* eps))
      out)))


;;; --------------------
;;; protocol: dropout
;;; --------------------

;;; Only uses mat-dimensions, mref, copy-mat, mgl-zeros, uniform-random!


(defmethod tb-dropout ((x mat) p &key training-p)
  "Dropout for MGL-MAT backend.

P is the drop probability in [0,1). When TRAINING-P is true, units are
zeroed with probability P and the remaining ones are scaled by 1/(1-P).
When TRAINING-P is NIL, returns a copy of X (no dropout applied)."
  (cond
    ;; no dropout when not training or p <= 0
    ((or (not training-p)
         (<= p 0.0d0))
     (copy-mat x))

    ;; extreme corner: p >= 1.0 -> all zeros
    ((>= p 1.0d0)
     (tb:tb-zeros :mgl (mat-dimensions x)))

    (t
     (let* ((dims (mat-dimensions x))
            (rows (first dims))
            (cols (second dims))
            (y    (copy-mat x))
            (mask (tb:tb-zeros :mgl dims)))
       ;; fill mask with uniform [0,1)
       (uniform-random! mask :limit 1.0d0)
       ;; threshold + scale
       (let ((scale (/ 1.0d0 (- 1.0d0 p))))
         (dotimes (i rows)
           (dotimes (j cols)
             (let ((r (mref mask i j)))
               (setf (mref mask i j)
                     (if (< r p)
                         0.0d0
                         scale))))))
       ;; apply mask: y := y * mask elementwise
       (dotimes (i rows)
         (dotimes (j cols)
           (setf (mref y i j)
                 (* (mref y i j) (mref mask i j)))))
       y))))


;;; FILE: src/mgl/package.lisp

(defpackage #:cl-transformer-blocks-mgl
  (:use #:cl #:cl-transformer-blocks #:mgl-mat)
  (:nicknames #:tb-mgl)
  (:import-from #:mgl-mat
                ;; type & construction
                #:mat
                #:make-mat
                #:mat-dimensions
                #:mat-size
                
                ;; element access / copying
                #:mref
                #:copy-mat
                
                ;; linear algebra
                #:mm*         ; matrix-matrix multiply
                #:axpy!       ; y := a*x + y
                #:scal!       ; x := a*x
                #:transpose
                
                ;; random & fill
                #:gaussian-random!
                #:uniform-random!
                #:fill!)
  (:export
   ;; Device/context control
   #:*device*
   #:with-cpu
   #:with-gpu

   ;; Helpers
   #:random-mat

   ;; Backend constructors
   #:make-attention-layer
   #:make-feedforward-layer
   #:make-block))


;;; FILE: src/mgl/second_ops.lisp

;;; src/mgl/ops.lisp

(in-package #:cl-transformer-blocks-mgl)

;;; ------------------------------------------------------------
;;; allocation & helpers
;;; ------------------------------------------------------------

(defmethod tb-zeros ((backend (eql :mgl)) dims &key (dtype :float))
  "Create a zero-initialized MGL-MAT:MAT of shape DIMS.
BACKEND is ignored except for dispatch."
  (declare (ignore backend dtype))
  (make-mat dims :ctype :float :initial-element 0.0d0))

(defun random-mat (rows cols &key (stddev 0.02d0))
  "Create a ROWS x COLS MGL-MAT:MAT with small Gaussian random values."
  (let ((m (tb:tb-zeros :mgl (list rows cols))))
    (gaussian-random! m :mean 0.0d0 :stddev stddev)
    m))

;;; ------------------------------------------------------------
;;; basic ops
;;; ------------------------------------------------------------

(defmethod tb-tensor-shape ((x mat))
  (mat-dimensions x))

(defmethod tb-matmul ((a mat) (b mat))
  (mm* a b))

(defmethod tb-add ((a mat) (b mat))
  (let ((y (copy-mat a)))
    (axpy! 1.0d0 b y)
    y))

(defmethod tb-add-scaled ((a mat) (b mat) scale)
  (let ((y (copy-mat a)))
    (axpy! scale b y)
    y))

(defmethod tb-scale ((x mat) alpha)
  (let ((y (copy-mat x)))
    (scal! alpha y)
    y))

(defmethod tb-transpose ((x mat))
  (transpose x))

;;; ------------------------------------------------------------
;;; GELU (approximate)
;;; ------------------------------------------------------------

(defun %gelu-approx (x)
  "Approximate GELU using tanh-based formula."
  (let* ((c (/ (sqrt (* 2.0d0 pi))))
         (inner (+ x (* 0.044715d0 x x x)))
         (tanh-arg (* c inner)))
    (* 0.5d0 x (+ 1.0d0 (tanh tanh-arg)))))

(defmethod tb-gelu ((x mat))
  (let* ((dims (mat-dimensions x))
         (rows (first dims))
         (cols (second dims))
         (y    (copy-mat x)))
    (dotimes (i rows)
      (dotimes (j cols)
        (let ((v (mref y i j)))
          (setf (mref y i j) (%gelu-approx v)))))
    y))

;;; ------------------------------------------------------------
;;; softmax (2D, axis = -1)
;;; ------------------------------------------------------------

(defun %softmax-row! (m row)
  "In-place softmax of ROW of M (2D MAT)."
  (let* ((dims (mat-dimensions m))
         (cols (second dims)))
    ;; max for stability
    (let ((max-val -1d300))
      (dotimes (j cols)
        (let ((v (mref m row j)))
          (when (> v max-val)
            (setf max-val v))))
      ;; exp shifted + sum
      (let ((sum 0d0))
        (dotimes (j cols)
          (let* ((v  (mref m row j))
                 (ev (exp (- v max-val))))
            (setf (mref m row j) ev)
            (incf sum ev)))
        ;; normalize
        (dotimes (j cols)
          (setf (mref m row j)
                (/ (mref m row j) sum)))))))

(defmethod tb-softmax ((x mat) &key (axis -1))
  (unless (eql axis -1)
    (error "tb-softmax (mgl): only AXIS = -1 supported, got ~S" axis))
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-softmax (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (y    (copy-mat x)))
      (dotimes (i rows)
        (%softmax-row! y i))
      y)))

;;; ------------------------------------------------------------
;;; layer-norm (2D, over last dim)
;;; ------------------------------------------------------------

(defun %layer-norm-row! (x out row gamma beta eps)
  "Layer-norm for one row of X into OUT."
  (let* ((dims (mat-dimensions x))
         (cols (second dims)))
    (let ((sum 0d0)
          (sumsq 0d0))
      (dotimes (j cols)
        (let ((v (mref x row j)))
          (incf sum v)
          (incf sumsq (* v v))))
      (let* ((cols-d (coerce cols 'double-float))
             (mean   (/ sum cols-d))
             (var    (max eps (- (/ sumsq cols-d) (* mean mean))))
             (inv-std (/ 1.0d0 (sqrt var))))
        (dotimes (j cols)
          (let* ((v    (mref x row j))
                 (norm (* (- v mean) inv-std))
                 (g    (if gamma
                           (mref gamma 0 j)
                           1.0d0))
                 (b    (if beta
                           (mref beta 0 j)
                           0.0d0))
                 (yval (+ (* norm g) b)))
            (setf (mref out row j) yval)))))))

(defmethod tb-layer-norm ((x mat) gamma beta &key (eps 1e-5))
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-layer-norm (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (cols (second dims))
           (gamma* (when gamma
                     (let ((gdims (mat-dimensions gamma)))
                       (unless (and (= (length gdims) 2)
                                    (= (first gdims) 1)
                                    (= (second gdims) cols))
                         (error "GAMMA must be (1 x ~D), got ~S" cols gdims))
                       gamma))
           (beta*  (when beta
                     (let ((bdims (mat-dimensions beta)))
                       (unless (and (= (length bdims) 2)
                                    (= (first bdims) 1)
                                    (= (second bdims) cols))
                         (error "BETA must be (1 x ~D), got ~S" cols bdims))
                       beta))
           (out (copy-mat x)))
      (dotimes (i rows)
        (%layer-norm-row! x out i gamma* beta* eps))
      out)))

;;; ------------------------------------------------------------
;;; dropout
;;; ------------------------------------------------------------

(defmethod tb-dropout ((x mat) p &key training-p)
  (cond
    ((or (not training-p)
         (<= p 0.0d0))
     (copy-mat x))
    ((>= p 1.0d0)
     (tb:tb-zeros :mgl (mat-dimensions x)))
    (t
     (let* ((dims (mat-dimensions x))
            (rows (first dims))
            (cols (second dims))
            (y    (copy-mat x))
            (mask (tb:tb-zeros :mgl dims))
            (scale (/ 1.0d0 (- 1.0d0 p))))
       ;; mask ~ U[0,1)
       (uniform-random! mask :limit 1.0d0)
       ;; threshold + scale
       (dotimes (i rows)
         (dotimes (j cols)
           (let ((r (mref mask i j)))
             (setf (mref mask i j)
                   (if (< r p) 0.0d0 scale)))))
       ;; apply mask element-wise
       (dotimes (i rows)
         (dotimes (j cols)
           (setf (mref y i j)
                 (* (mref y i j) (mref mask i j)))))
       y))))


;;; FILE: tests/mgl-tests.lisp

(in-package #:cl-transformer-blocks.tests)

(in-suite :cl-transformer-blocks/mgl)

;;; ------------------------------------------------------------
;;; basic tensor tests
;;; ------------------------------------------------------------

(test zeros-shape
  (let* ((dims '(3 4))
         (x    (tb-zeros :mgl dims)))
    (is (equal dims (tb-tensor-shape x)))))

(test random-mat-shape
  (let* ((rows 5)
         (cols 7)
         (x    (random-mat rows cols)))
    (is (equal (list rows cols)
               (tb-tensor-shape x)))))

;;; ------------------------------------------------------------
;;; single block forward
;;; ------------------------------------------------------------

(test single-block-forward-shape
  (let* ((d          32)
         (time-steps 10)
         (x          (random-mat d time-steps))
         ;; MGL-backed transformer block
         (blk        (make-block d))
         (y          (forward blk x)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))))

;;; ------------------------------------------------------------
;;; stack of blocks
;;; ------------------------------------------------------------

(test stacked-blocks-forward-shape
  (let* ((d          32)
         (time-steps 10)
         (x          (random-mat d time-steps))
         (blocks     (loop repeat 3 collect (make-block d)))
         (stack      (make-block-list blocks))
         (y          (forward stack x)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))))

;;; ------------------------------------------------------------
;;; layer norm
;;; ------------------------------------------------------------

(test layer-norm-shape
  (let* ((rows 4)
         (cols 8)
         (x   (random-mat rows cols))
         ;; gamma/beta NIL: backend uses its default parameters
         (y   (tb-layer-norm x nil nil)))
    ;; shape invariant
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))
    ;; semantic check: per-feature mean ≈ 0, variance ≈ 1
    (destructuring-bind (r c) (tb-tensor-shape y)
      (dotimes (j c)
        (let ((sum 0d0)
              (sqsum 0d0))
          (dotimes (i r)
            (let ((v (mref y i j)))
              (incf sum v)
              (incf sqsum (* v v))))
          (let* ((mean (/ sum r))
                 (var  (- (/ sqsum r) (* mean mean))))
            (is (< (abs mean) 1d-6))
            (is (< (abs (- var 1d0)) 1d-3))))))))

;;; ------------------------------------------------------------
;;; dropout
;;; ------------------------------------------------------------

(test dropout-shape-and-mode
  (let* ((rows 32)
         (cols 16)
         (p    0.5d0)
         (x       (random-mat rows cols))
         (y-train (tb-dropout x p :training-p t))
         (y-test  (tb-dropout x p :training-p nil)))
    ;; shapes stay the same
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y-train)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y-test)))

    ;; inference mode: should be (numerically) identical to x
    (let ((diff-sum 0d0))
      (destructuring-bind (r c) (tb-tensor-shape x)
        (dotimes (i r)
          (dotimes (j c)
            (incf diff-sum
                  (abs (- (mref y-test i j)
                          (mref x i j)))))))
      (is (< diff-sum 1d-12)))

    ;; training mode: we expect roughly a fraction P of entries to be zeroed
    (let ((zeros 0)
          (total 0))
      (destructuring-bind (r c) (tb-tensor-shape y-train)
        (dotimes (i r)
          (dotimes (j c)
            (incf total)
            (when (zerop (mref y-train i j))
              (incf zeros))))
        (let* ((frac (if (plusp total)
                         (/ zeros total)
                         0d0)))
          ;; Very loose tolerance to avoid flakes; we just want to know
          ;; dropout is doing *something* like the right probability.
          (is (< (abs (- frac p)) 0.25d0)))))))


;;; FILE: tests/package.lisp

(defpackage #:cl-transformer-blocks.tests
  (:use #:cl
        #:fiveam
        ;; We are testing the MGL backend, so we also pull in mgl-mat
        #:mgl-mat)
  (:nicknames #:tb-tests)
  (:import-from #:cl-transformer-blocks
    ;; core protocol + containers
    #:tb-zeros
    #:tb-tensor-shape
    #:tb-layer-norm
    #:tb-dropout
    #:forward
    #:make-block-list)
  (:import-from #:cl-transformer-blocks-mgl
    ;; backend-specific helpers
    #:random-mat
    #:make-block))

(in-package #:cl-transformer-blocks.tests)

;; Main suite for this project / backend
(def-suite :cl-transformer-blocks/mgl)

(defun run-tests ()
  "Run all cl-transformer-blocks tests for the MGL backend."
  (fiveam:run! :cl-transformer-blocks/mgl))


