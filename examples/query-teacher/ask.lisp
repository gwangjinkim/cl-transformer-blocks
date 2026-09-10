;; sbcl --no-sysinit --no-userinit --script examples/query-teacher/ask.lisp \
;;   .build/query-teacher-metal/merged "Show open tokenizer issues from the last week." gpu
(load "scripts/load.lisp")
(load "examples/query-teacher/domain.lisp")
(in-package #:tb-query-teacher)

(destructuring-bind (directory request &optional (device "cpu")) (uiop:command-line-arguments)
  (unless (member device '("cpu" "gpu") :test #'equal)
    (error "Device must be cpu or gpu"))
  (tb:with-resource (model (tb:from-pretrained directory :device (if (equal device "gpu") :gpu :cpu)))
    (let* ((prompt (tb:encode-text model (prompt-for request) :add-special-tokens nil))
           (tokens (tb:generate model prompt :max-new-tokens 24 :eos-token-id 0))
           (answer (tb:decode-tokens model (subseq tokens (length prompt)) :skip-special-tokens t))
           (query (parse-query answer)))
      (format t "~&Model: ~A~%" answer)
      (if query
          (format t "Matching example issue IDs: ~S~%" (execute-query query (example-issues)))
          (error "Model output was rejected by the application's command validator")))))
