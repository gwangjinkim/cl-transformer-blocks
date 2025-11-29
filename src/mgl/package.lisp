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
