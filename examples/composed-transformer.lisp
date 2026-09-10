;; From the repository root after bootstrap. Random initialization, not a pretrained LLM.
(load "scripts/load.lisp")
(let* ((attention (tb:make-rotary-attention :heads 4 :kv-heads 2))
       (blocks (list
                (tb:make-transformer-block :attention attention
                  :feed-forward (tb:make-swiglu :intermediate-size 28)
                  :residual :sequential)
                (tb:make-transformer-block
                  :attention (tb:make-rotary-attention :heads 2 :kv-heads 1)
                  :feed-forward (tb:make-swiglu :intermediate-size 36)
                  :residual :parallel))))
  (tb:with-resource (model (tb:make-transformer :vocab-size 32 :hidden-size 16 :blocks blocks
                            :context-length 64 :seed 17
                            :device (if (equal "gpu" (uiop:getenv "TB_DEVICE")) :gpu :cpu)))
    (format t "~&Composed ~D blocks with ~D named parameter tensors.~%"
            (length (tb:transformer-blocks model)) (length (tb:named-parameters model)))
    (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
      (dotimes (step 3)
        (format t "Step ~D loss: ~,5F~%" (1+ step)
                (tb:train-step model optimizer #2A((3 4 5 6)) :max-grad-norm 1.0))))
    (tb:save-pretrained model ".build/composed-example/")
    (format t "Exported .build/composed-example/ for native Lisp or Python AutoClass loading.~%")))
