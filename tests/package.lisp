(defpackage #:cl-transformer-blocks.tests
  (:use #:cl
        #:fiveam
        #:cl-transformer-blocks
        #:cl-transformer-blocks-mgl)
  (:nicknames #:tb-tests))

(in-package #:cl-transformer-blocks.tests)

;; Define a main suite for the project
(def-suite :cl-transformer-blocks/mgl)

(defun run-tests ()
  "Run all cl-transformer-blocks tests."
  (fiveam:run! :cl-transformer-blocks/mgl))
