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
