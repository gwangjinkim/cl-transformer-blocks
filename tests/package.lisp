(defpackage #:cl-transformer-blocks.tests
  (:use #:cl
        #:fiveam
        #:cl-transformer-blocks
        #:cl-transformer-blocks-mgl)
  (:nicknames #:tb-tests))

(in-package #:cl-transformer-blocks.tests)

;; Main suite for all tests in this project
(def-suite :cl-transformer-blocks/mgl
  :description "Tests for the MGL backend of cl-transformer-blocks.")

(defun run-tests ()
  "Run all cl-transformer-blocks tests."
  (fiveam:run! :cl-transformer-blocks/mgl))
