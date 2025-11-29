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
   
   ;; Core API
   #:forward
   #:transformer-block
   #:block-list

   ;; Layer classes (for backends to construct)
   #:attention-layer
   #:feedforward-layer))
