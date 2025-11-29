(in-package #:cl-transformer-blocks-mgl)

;;; MGL-based attention forward

(defmethod cl-transformer-blocks:forward
    ((layer cl-transformer-blocks:attention-layer)
     (x mgl-mat:mat)
     &key mask training-p)
  (declare (ignore mask training-p))
  (let* ((w-q (cl-transformer-blocks:attention-w-q layer))
         (w-k (cl-transformer-blocks:attention-w-k layer))
         (w-v (cl-transformer-blocks:attention-w-v layer))
         (w-o (cl-transformer-blocks:attention-w-o layer))

         (q (cl-transformer-blocks:tb-matmul w-q x))
         (k (cl-transformer-blocks:tb-matmul w-k x))
         (v (cl-transformer-blocks:tb-matmul w-v x))

         (scores  (cl-transformer-blocks:tb-matmul
                   (cl-transformer-blocks:tb-transpose k) q))
         (weights (cl-transformer-blocks:tb-softmax scores))
         (ctx     (cl-transformer-blocks:tb-matmul v weights)))
    (cl-transformer-blocks:tb-matmul w-o ctx)))

;;; MGL-based feedforward forward

(defmethod cl-transformer-blocks:forward
    ((layer cl-transformer-blocks:feedforward-layer)
     (x mgl-mat:mat)
     &key mask training-p)
  (declare (ignore mask))
  (let* ((w1 (cl-transformer-blocks:ffn-w1 layer))
         (w2 (cl-transformer-blocks:ffn-w2 layer))
         (h  (cl-transformer-blocks:tb-gelu
              (cl-transformer-blocks:tb-matmul w1 x)))
         (out (cl-transformer-blocks:tb-matmul w2 h)))
    (cl-transformer-blocks:tb-dropout out 0.0d0 :training-p training-p)))
