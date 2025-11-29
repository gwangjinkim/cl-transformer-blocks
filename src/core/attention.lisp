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
