(in-package #:tb)
(define-condition backend-error (error)
  ((message :initarg :message :reader error-message))
  (:report (lambda (c s) (write-string (error-message c) s))))
(define-condition compatibility-error (backend-error) ())
(define-condition shape-error (backend-error) ())
(defun fail (type format-control &rest args)
  (error type :message (apply #'format nil format-control args)))
(defgeneric dispose (object) (:documentation "Release owned native resources; idempotent."))
(defmethod dispose ((object null)) nil)
(defmacro with-resource ((name expression) &body body)
  "Evaluate EXPRESSION once and release its resource on every exit."
  `(let ((,name ,expression)) (unwind-protect (progn ,@body) (dispose ,name))))
(defgeneric tensor-shape (tensor))
(defgeneric tensor-dtype (tensor))
(defgeneric tensor-array (tensor))
(defgeneric forward (module inputs &key cache attention-mask token-type-ids)
  (:documentation "Return owned logits. INPUTS have shape (batch,time).
TOKEN-TYPE-IDS supplies matching segment IDs for native BERT or the Python model."))
(defgeneric named-parameters (module))
(defgeneric save-pretrained (model destination &key max-shard-size))
(defgeneric python-architecture-files (model)
  (:documentation "Return emitted (filename . source-pathname) Python architecture files."))
(defgeneric model-capabilities (model)
  (:documentation "Return a property list describing the loaded model's actual execution contract."))
(defgeneric architecture-capabilities (architecture config directory)
  (:documentation "Return architecture-specific operations and restrictions without loading weights."))
(defgeneric make-cache (model &key growth-step))
(defgeneric encode-text (model text &key add-special-tokens))
(defgeneric encode-batch (model texts &key text-pairs add-special-tokens padding max-length
                                         truncation pad-token-id)
  (:documentation "Return rank-two input IDs, attention masks, and token-type IDs as three values.
Native tokenizer.json execution; right padding and optional right longest-first truncation."))
(defgeneric attach-tokenizer (model directory)
  (:documentation "Snapshot tokenizer assets and replace a native model's tokenizer after validation."))
(defgeneric decode-tokens (model ids &key skip-special-tokens))
(defgeneric generate (model prompt &key max-new-tokens eos-token-id))
(defgeneric load-python-model (source &key device revision task local-files-only
                                           trust-remote-code dtype python-executable
                                           max-output-elements auto-class python-threads
                                           training-dtype))
(defgeneric python-forward (model inputs &key outputs transport))
(defgeneric python-process (model inputs &key options outputs))
(defgeneric python-processor-forward (model inputs &key options model-inputs outputs))
(defgeneric python-generate (model inputs &key options outputs seed))
(defgeneric python-processor-generate
    (model inputs &key processor-options options model-inputs outputs seed))
(defgeneric python-apply-chat-template
    (model messages &key add-generation-prompt continue-final-message tokenize options))
(defgeneric python-train-step (model optimizer inputs &key max-grad-norm))
(defgeneric python-processor-train-step
    (model optimizer inputs &key options model-inputs max-grad-norm))
(defgeneric train-step (model optimizer ids &key labels attention-mask token-type-ids max-grad-norm))
(defgeneric prepare-training-batch (model ids labels mask))
(defgeneric python-train-microbatches (model optimizer microbatches &key max-grad-norm))
(defgeneric configure-python-scheduler
    (model optimizer kind &key warmup-steps total-steps scheduler-options))
(defgeneric python-optimizer-info (model optimizer))
(defgeneric python-adapter-info (model &key adapter-name))
(defgeneric python-make-lora
    (model &key rank alpha target-modules dropout rslora use-dora bias task-type
                modules-to-save adapter-name))
(defgeneric python-load-adapter
    (model source &key adapter-name trainable revision local-files-only))
(defgeneric python-save-adapter (model destination &key adapter-name))
(defgeneric python-merge-adapter (model &key safe-merge adapter-names))
(defgeneric save-training-checkpoint (model optimizer destination))
(defgeneric restore-training-checkpoint (model optimizer source &key restore-rng))
(defgeneric push-to-hub
    (model repo-id &key private revision commit-message commit-description create-pr
                       model-card dry-run-directory))
(defun valid-hub-repo-id-p (repo-id)
  (when (and (stringp repo-id) (<= 1 (length repo-id) 96)
             (<= (count #\/ repo-id) 1)
             (not (search "--" repo-id)) (not (search ".." repo-id))
             (every (lambda (character)
                      (or (alphanumericp character) (find character "-_.\/")))
                    repo-id))
    (let ((parts (uiop:split-string repo-id :separator '(#\/))))
      (every (lambda (part)
               (and (plusp (length part))
                    (not (member (char part 0) '(#\. #\-)))
                    (not (member (char part (1- (length part))) '(#\. #\-)))))
             parts))))
(defun default-python-executable ()
  (or (uiop:getenv "TB_PYTHON")
      (let ((local (merge-pathnames #+windows ".venv/Scripts/python.exe"
                                    #-windows ".venv/bin/python"
                                    (asdf:system-source-directory "cl-transformer-blocks"))))
        (and (probe-file local) (namestring local)))
      "python3"))
