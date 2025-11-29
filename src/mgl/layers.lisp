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
