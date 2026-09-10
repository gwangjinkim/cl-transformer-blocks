;; Requires the cached checkpoint used by PRETRAINED-COMPONENTS.LISP.
;; Rebuild the edited base deterministically before creating its adapter.
(load "examples/recompose-pretrained.lisp")

(defun distillation-example-ids (model text)
  (let* ((tokens (tb:encode-text model text :add-special-tokens nil))
         (count (min 24 (length tokens)))
         (ids (make-array (list 1 count))))
    (dotimes (i count ids) (setf (aref ids 0 i) (aref tokens i)))))

(defun distillation-example-loss (student dataset)
  (/ (loop for index below (tb:distillation-dataset-size dataset) sum
       (tb:with-resource (target (tb:load-distillation-dataset-batch dataset index))
         (multiple-value-bind (loss gradients)
             (tb:distillation-batch-loss-and-gradients student target :temperature 2.0)
           (unwind-protect loss
             (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients)))))
     (tb:distillation-dataset-size dataset)))

(let* ((device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
       (source (or (uiop:getenv "TB_MODEL") ".build/models/smollm2/"))
       (base ".build/recomposed-example/")
       (output (uiop:ensure-directory-pathname ".build/distillation-example/"))
       (train '("Common Lisp programs can define classes, methods, and reusable functions."
                "A transformer predicts the next token using the preceding text as context."
                "The library loads model weights and performs numerical calculations on the GPU."
                "We test an exported model in Python to verify that its predictions agree."))
       (held '("A Lisp function accepts arguments and returns values to its caller."
               "Training adjusts model parameters to reduce an objective measured on examples.")))
  (ensure-directories-exist output)
  (tb:with-resource (student (tb:from-pretrained base :device device))
    (tb:make-lora student :rank 4 :alpha 8 :seed 37)
    (let* ((train-batches (mapcar (lambda (text) (distillation-example-ids student text)) train))
           (held-batches (mapcar (lambda (text) (distillation-example-ids student text)) held))
           (train-directory (merge-pathnames "train-targets/" output))
           (held-directory (merge-pathnames "held-targets/" output))
           (train-examples (mapcar #'tb:make-distillation-example train-batches))
           (held-examples (mapcar #'tb:make-distillation-example held-batches)))
      (tb:with-resource (teacher (tb:from-pretrained source :device device))
        (tb:save-distillation-dataset-from-teacher
         teacher train-examples train-directory :dataset-id "smollm2-demo-train-v1")
        (tb:save-distillation-dataset-from-teacher
         teacher held-examples held-directory :dataset-id "smollm2-demo-held-v1"))
      ;; The full teacher is gone. Production and training retain only one
      ;; target at a time and use a portable deterministic shuffle.
      (let* ((train-dataset (tb:load-distillation-dataset
                             train-directory :device device :shuffle t :seed 37))
             (held-dataset (tb:load-distillation-dataset held-directory :device device))
             (before-train (distillation-example-loss student train-dataset))
             (before-held (distillation-example-loss student held-dataset)))
        (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.0005 :weight-decay 0.0))
          (dotimes (epoch 5)
            (tb:start-distillation-dataset-epoch train-dataset epoch)
            (loop while (tb:distill-dataset-step student train-dataset optimizer
                                                 :temperature 2.0 :max-grad-norm 1.0)))
          (tb:save-training-checkpoint student optimizer (merge-pathnames "checkpoint/" output))
          (tb:save-distillation-dataset-state
           train-dataset (merge-pathnames "dataset-state/" output)))
        (let ((after-train (distillation-example-loss student train-dataset))
              (after-held (distillation-example-loss student held-dataset)))
          (tb:save-adapter student (merge-pathnames "adapter/" output))
          (tb:with-resource (logits (tb:forward student (first held-batches)))
            (tb::save-weights (list (cons "logits" logits))
                              (merge-pathnames "native-logits.safetensors" output)))
          (tb::write-json
           (tb::component-object
            "teacher" (namestring (truename source)) "student" (namestring (truename base))
            "device" (string-downcase (symbol-name device)) "temperature" 2.0
            "steps" 20 "seed" 37 "shuffle" 'yason:true
            "train_dataset_id" "smollm2-demo-train-v1"
            "held_dataset_id" "smollm2-demo-held-v1"
            "train_texts" (coerce train 'vector) "held_texts" (coerce held 'vector)
            "train_before" before-train "train_after" after-train
            "held_before" before-held "held_after" after-held)
           (merge-pathnames "report.json" output))
          (format t "~&Mean T^2 KL, training: ~,6F -> ~,6F; held out: ~,6F -> ~,6F~%"
                  before-train after-train before-held after-held)))
      (format t "~&~A~%" (tb:decode-tokens student
                          (tb:generate
                           student (tb:encode-text student "Common Lisp is a programming language"
                                                   :add-special-tokens nil)
                           :max-new-tokens 16)
                          :skip-special-tokens t))
      (format t "Saved lazy train/held target datasets, iterator state, adapter and optimizer checkpoint. Six short texts cannot establish language quality.~%"))))
