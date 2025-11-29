(in-package #:cl-transformer-blocks-mgl)

;;; ------------------------------------------------------------
;;; MGL-backed FORWARD implementations for the core layer types
;;; ------------------------------------------------------------

;; We assume:
;;  - ATTENTION-LAYER and FEEDFORWARD-LAYER are the core classes defined
;;    in the CL-TRANSFORMER-BLOCKS package (with slots W-Q, W-K, W-V,
;;    W-O, W1, W2, etc.).
;;  - MAT is MGL-MAT:MAT, imported or :USED in this package.
;;  - TB-MATMUL and TB-GELU are already implemented for MAT in ops.lisp.

(defmethod forward ((layer attention-layer) (x mat))
  "Backend-specific FORWARD for ATTENTION-LAYER on MGL-MAT:MAT.

Currently a very simple placeholder:
  y = W_O * x

This preserves shape and lets us test block wiring. Later, you can
replace this with full (multi-head) scaled dot-product self-attention
using W-Q, W-K and W-V as well."
  (let ((w-o (cl-transformer-blocks:attention-w-o layer)))
    (tb-matmul w-o x))

(defmethod forward ((layer cl-transformer-blocks:attention-layer)
                    (x mat))
  "Backend-specific FORWARD for ATTENTION-LAYER on MGL-MAT:MAT.

Placeholder: y = W_O * x."
  (let ((w-o (cl-transformer-blocks:attention-w-o layer)))
    (tb-matmul w-o x)))

(defmethod forward ((layer cl-transformer-blocks:feedforward-layer)
                    (x mat))
  "Backend-specific FORWARD for FEEDFORWARD-LAYER on MGL-MAT:MAT.

Standard FFN: y = W2 * GELU(W1 * x)."
  (let* ((w1 (cl-transformer-blocks:ffn-w1 layer))
         (w2 (cl-transformer-blocks:ffn-w2 layer))
         (h  (tb-gelu (tb-matmul w1 x))))
    (tb-matmul w2 h)))



  ;; NOTE: If you want to see something more realistic later, you'll do:
  ;;   Q = W_Q * X
  ;;   K = W_K * X
  ;;   V = W_V * X
  ;;   scores = softmax( (Q^T K) / sqrt(d_k) )
  ;;   Y = W_O * (V * scores^T)
  ;; but we'll keep the placeholder for now.
  )

(defmethod forward ((layer cl-transformer-blocks:feedforward-layer) (x mgl-mat:mat))
  "Backend-specific FORWARD for FEEDFORWARD-LAYER on MGL-MAT:MAT.

Standard Transformer FFN:
  h = GELU(W1 * x)
  y = W2 * h

X is expected to be of shape [d_model, T]."
  (let* ((w1 (cl-transformer-blocks:ffn-w1 layer))
         (w2 (cl-transformer-blocks:ffn-w2 layer))
         (h  (tb-gelu (tb-matmul w1 x))))
    (tb-matmul w2 h)))
