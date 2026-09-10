(in-package #:tb-query-teacher)

(defun object (&rest pairs)
  (let ((table (make-hash-table :test 'equal)))
    (loop for (key value) on pairs by #'cddr do (setf (gethash key table) value))
    table))

(defun seconds-since (start)
  (/ (- (get-internal-real-time) start) (float internal-time-units-per-second 1d0)))

(defun epoch-order (count epoch)
  "Specified Fisher-Yates shuffle, independent of implementation RANDOM state."
  (let ((order (coerce (loop for i below count collect i) 'vector))
        (state (+ 17 epoch)))
    (loop for i downfrom (1- count) above 0 do
      (setf state (mod (+ (* 1664525 state) 1013904223) (expt 2 32)))
      (rotatef (aref order i) (aref order (mod state (1+ i)))))
    (coerce order 'list)))

(defun training-batch (encoded)
  "ENCODED contains (prompt-IDs answer-and-EOS-IDs); right-pad with token 0."
  (let* ((width (loop for (prompt answer) in encoded maximize (+ (length prompt) (length answer))))
         (shape (list (length encoded) width))
         (ids (make-array shape :initial-element 0))
         (mask (make-array shape :initial-element 0))
         (labels (make-array shape :initial-element -100)))
    (loop for (prompt answer) in encoded for row from 0 do
      (loop for token across prompt for column from 0 do
        (setf (aref ids row column) token (aref mask row column) 1))
      (loop for token across answer for column from (length prompt) do
        (setf (aref ids row column) token (aref mask row column) 1
              (aref labels row column) token)))
    (values ids mask labels)))

(defun encode-sample (model sample)
  (list (tb:encode-text model (prompt-for (sample-request sample)) :add-special-tokens nil)
        (concatenate 'vector
                     (tb:encode-text model (format nil " ~A" (render-query (sample-query sample)))
                                     :add-special-tokens nil)
                     #(0))))

(defun sample-json (sample)
  (object "request" (sample-request sample) "prompt" (prompt-for (sample-request sample))
          "target" (render-query (sample-query sample))
          "split" (string-downcase (symbol-name (sample-split sample)))
          "held_combination" (if (sample-held-combination-p sample) 'yason:true 'yason:false)
          "expected_issue_ids" (coerce (execute-query (sample-query sample) (example-issues)) 'vector)))

(defun evaluate-samples (model samples)
  (let ((start (get-internal-real-time)))
    (loop for sample in samples for index from 1 collect
      (let* ((prompt (tb:encode-text model (prompt-for (sample-request sample)) :add-special-tokens nil))
             (generated (tb:generate model prompt :max-new-tokens 24 :eos-token-id 0))
             (new-ids (subseq generated (length prompt)))
             (text (tb:decode-tokens model new-ids :skip-special-tokens t))
             (query (parse-query text))
             (correct (and query (equalp query (sample-query sample)))))
        (format t "~&eval ~D/~D ~A => ~S~%" index (length samples)
                (if correct "correct" "wrong") text)
        (finish-output)
        (object "request" (sample-request sample) "target" (render-query (sample-query sample))
                "prompt_ids" prompt "generated_ids" new-ids "prediction" text
                "split" (string-downcase (symbol-name (sample-split sample)))
                "held_combination" (if (sample-held-combination-p sample) 'yason:true 'yason:false)
                "valid" (if query 'yason:true 'yason:false)
                "correct" (if correct 'yason:true 'yason:false)
                "issue_ids" (coerce (when query (execute-query query (example-issues))) 'vector)
                "elapsed_cumulative_seconds" (seconds-since start))))))

(defun save-probes (model samples output)
  ;; Full teacher-forced logits check more than matching a handful of argmax IDs.
  (loop for sample in samples for i below 2 do
    (destructuring-bind (prompt answer) (encode-sample model sample)
      (let* ((tokens (concatenate 'vector prompt answer))
             (ids (make-array (list 1 (length tokens)) :initial-contents (list tokens))))
        (tb::write-json tokens (merge-pathnames (format nil "probe-~D.json" i) output))
        (tb:with-resource (logits (tb:forward model ids))
          (tb::save-weights (list (cons "logits" logits))
                           (merge-pathnames (format nil "probe-~D.safetensors" i) output)))))))

(defun run-demo (source output device epochs batch-size smoke)
  (let* ((data (make-curriculum))
         (train (remove-if-not (lambda (x) (eq :train (sample-split x))) data))
         (evaluation (remove-if (lambda (x) (eq :train (sample-split x))) data))
         (evaluation (if smoke (subseq evaluation 0 2) evaluation))
         (start (get-internal-real-time)))
    (ensure-directories-exist (merge-pathnames "dataset.json" output))
    (tb::write-json (coerce (mapcar #'sample-json data) 'vector)
                   (merge-pathnames "dataset.json" output))
    (tb:with-resource (model (tb:from-pretrained source :device device
                              :model-id "HuggingFaceTB/SmolLM2-135M"
                              :revision "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"))
      (format t "~&Loaded model: ~S~%" (tb:model-capabilities model))
      (let* ((encoded (coerce (mapcar (lambda (x) (encode-sample model x)) train) 'vector))
             (baseline (evaluate-samples model evaluation))
             (losses nil) (training-start (get-internal-real-time)))
        (tb::write-json (coerce baseline 'vector) (merge-pathnames "baseline.json" output))
        (tb:make-lora model :rank 8 :alpha 16 :seed 17
                      :targets '("q_proj" "v_proj" "o_proj" "gate_proj" "up_proj" "down_proj"))
        (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001 :weight-decay 0.01))
          (loop for epoch from 1 to epochs do
            (loop for remaining on (epoch-order (length encoded) epoch) by
                    (lambda (rest) (nthcdr (min batch-size (length rest)) rest))
                  for step from 1 do
              (multiple-value-bind (ids mask labels)
                  (training-batch
                   (mapcar (lambda (i) (aref encoded i))
                           (subseq remaining 0 (min batch-size (length remaining)))))
                (let ((loss (tb:train-step model optimizer ids :labels labels
                                           :attention-mask mask :max-grad-norm 1.0)))
                  (push loss losses)
                  (format t "~&epoch ~D step ~D loss ~,5F (~,1Fs)~%"
                          epoch step loss (seconds-since training-start))
                  (finish-output)))
              (when smoke (return))))
          (tb:save-training-checkpoint model optimizer (merge-pathnames "checkpoint/" output)))
        (let ((training-seconds (seconds-since training-start)))
          (tb::write-json
           (object "device" (string-downcase (symbol-name device))
                   "smoke" (if smoke 'yason:true 'yason:false)
                   "epochs" epochs "batch_size" batch-size "learning_rate" 0.001
                   "rank" 8 "alpha" 16 "seed" 17 "max_new_tokens" 24
                   "training_seconds" training-seconds "losses" (coerce (nreverse losses) 'vector)
                   "backend_memory" (princ-to-string (tb:backend-memory (tb:model-backend model))))
           (merge-pathnames "training.json" output)))
        (tb::write-json (coerce (evaluate-samples model evaluation) 'vector)
                       (merge-pathnames "adapted.json" output))
        (save-probes model evaluation output)
        (tb:save-adapter model (merge-pathnames "adapter/" output))
        (tb:merge-adapter model)
        (tb:save-pretrained model (merge-pathnames "merged/" output))
        (format t "~&Native workflow completed in ~,1Fs. Python verification is next.~%"
                (seconds-since start))))))
