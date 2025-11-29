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
