;;; Real transformer block: attention + FFN + residual connections (no LN yet, you can add later once backend supports it).

(in-package #:cl-transformer-blocks)

(defclass transformer-block ()
  ((attention :initarg :attention :reader block-attention)
   (ffn       :initarg :ffn       :reader block-ffn))
  (:documentation
   "A single transformer block: self-attention + feed-forward + residuals.

This implementation currently omits layer normalization to keep the
backend implementation simple; you can add LNs once TB-LAYER-NORM is
implemented in your backend."))

(defun %check-block-slots (blk)
  (unless (and (slot-boundp blk 'attention)
               (slot-boundp blk 'ffn))
    (error "TRANSFORMER-BLOCK is missing ATTENTION and/or FFN sublayers.
Use a backend-specific constructor (e.g. TB-MGL:MAKE-BLOCK).")))

(defmethod forward ((blk transformer-block) x &key mask training-p)
  (%check-block-slots blk)
  (let* ((attn-layer (block-attention blk))
         (ffn-layer  (block-ffn blk))

         ;; Self-attention + residual
         (attn-out (forward attn-layer x :mask mask :training-p training-p))
         (res1     (tb-add x attn-out))

         ;; Feed-forward + residual
         (ffn-out  (forward ffn-layer res1 :training-p training-p))
         (res2     (tb-add res1 ffn-out)))
    res2))
