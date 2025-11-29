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
        
(defmethod forward ((blk transformer-block) x &rest args &key &allow-other-keys)
  "Forward pass through a Transformer block using the TB-* protocol.

Pre-norm variant if BLOCK-USE-LAYER-NORM is true."
  (declare (ignore args))
  (labels ((maybe-ln (tensor)
             (if (block-use-layer-norm blk)
                 ;; gamma / beta NIL: backend can use defaults
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
  "Forward pass through a stack of blocks.

Applies each block's FORWARD in sequence:

  x_0 = x
  x_{k+1} = FORWARD(block_k, x_k)

Returns the output of the final block."
  (reduce #'forward (block-list-blocks stack)
          :initial-value x))
