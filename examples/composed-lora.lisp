;; Run from the repository root after bootstrap. This is a tiny mechanics example,
;; not a pretrained language model. First construct, train and save the base.
(load "examples/composed-transformer.lisp")

(tb:with-resource
    (model (tb:from-pretrained ".build/composed-example/"
                              :device (if (equal "gpu" (uiop:getenv "TB_DEVICE")) :gpu :cpu)))
  ;; Reloading the saved base gives the PEFT adapter a reusable base identity.
  (tb:make-lora model :rank 3 :alpha 6 :rslora t :seed 37
                :targets '("q_proj" "v_proj" "down_proj"))
  (format t "~&Training ~D adapter tensors while ~D base tensors stay frozen.~%"
          (length (tb:trainable-parameters model)) (length (tb:named-parameters model)))
  (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
    (dotimes (step 3)
      (format t "Adapter step ~D loss: ~,5F~%" (1+ step)
              (tb:train-step model optimizer #2A((3 4 5 6))
                             :labels #2A((-100 -100 5 6)) :max-grad-norm 1.0)))
    (tb:save-training-checkpoint model optimizer ".build/composed-lora-checkpoint/"))
  (tb:save-adapter model ".build/composed-lora-adapter/")
  (tb:merge-adapter model)
  (tb:save-pretrained model ".build/composed-lora-merged/")
  (format t "Saved a PEFT adapter, resumable checkpoint, and standalone merged model.~%"))
