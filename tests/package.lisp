(defpackage #:cl-transformer-blocks.tests
  (:use #:cl
        #:fiveam
        ;; We are testing the MGL backend, so we also pull in mgl-mat
        #:mgl-mat)
  (:nicknames #:tb-tests)
  (:import-from #:cl-transformer-blocks
    ;; core protocol + containers
    #:tb-zeros
    #:tb-tensor-shape
    #:tb-layer-norm
    #:tb-dropout
    #:forward
    #:make-block-list)
  (:import-from #:cl-transformer-blocks-mgl
    ;; backend-specific helpers
    #:random-mat
    #:make-block))

(in-package #:cl-transformer-blocks.tests)

;; Main suite for this project / backend
(def-suite :cl-transformer-blocks/mgl)

(defun run-tests ()
  "Run all cl-transformer-blocks tests for the MGL backend."
  (fiveam:run! :cl-transformer-blocks/mgl))
