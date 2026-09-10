(in-package #:tb-tests)

(defun run-python-worker-tests ()
  (let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
         (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (fixture (merge-pathnames ".build/fixtures/python-worker-opt/" root))
         (exported (merge-pathnames (format nil ".build/python-worker-export-~(~A~)/" device) root))
         (sharded (merge-pathnames
                   (format nil ".build/python-worker-sharded-~(~A~)/" device) root))
         (reference (tb::read-json (merge-pathnames "worker-reference.json" fixture)))
         (ids (json-array (gethash "input_ids" reference)))
         (mask (json-array (gethash "attention_mask" reference)))
         (expected (json-array (gethash "logits" reference))))
    (signals tb:compatibility-error (tb:from-pretrained fixture))
    (signals tb:compatibility-error
      (tb:from-pretrained fixture :execution :python :local-files-only t
                                  :auto-class "AutoTokenizer"))
    (let ((model (tb:from-pretrained fixture :execution :python :device device
                                     :local-files-only t :max-output-elements 300)))
      (unwind-protect
           (progn
             (check (typep model 'tb:python-model) "explicit Python model type")
             (check (tb:python-worker-alive-p model) "worker remains resident")
             (check (equal (gethash "model_type" (tb:model-config model)) "opt")
                    "worker preserves arbitrary Transformers config")
             (check (equal (gethash "requested_device" (tb:python-worker-info model))
                           (string-downcase device)) "worker records requested device")
             (check (if (eq device :cpu)
                        (equal (gethash "actual_device" (tb:python-worker-info model)) "cpu")
                        (member (gethash "actual_device" (tb:python-worker-info model))
                                '("mps" "cuda") :test #'equal))
                    "worker proves actual execution device without fallback")
             (let ((capabilities (tb:model-capabilities model)))
               (check (getf capabilities :supported-p) "loaded Python model is supported")
               (check (eq (getf capabilities :execution) :python)
                      "worker capability reports Python execution")
               (check (eq (getf capabilities :backend) :pytorch)
                      "worker capability reports PyTorch backend")
               (check (equal (getf capabilities :model-type) "opt")
                      "worker capability preserves arbitrary model type")
               (check (member (getf capabilities :device) '(:cpu :mps :cuda))
                      "worker capability reports actual device")
               (check (eq (getf capabilities :requested-device) device)
                      "worker capability reports requested device")
               (check (equal (getf capabilities :execution-dtype) "float32")
                      "worker capability reports actual model dtype")
               (check (every (lambda (operation)
                               (member operation (getf capabilities :operations)))
                             '(:forward :training :generation :processor
                               :training-checkpoint :export :hub-push))
                      "worker operations are discoverable"))
             (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
               (check (equal (tb:tensor-shape logits) (array-dimensions expected))
                      "Python tensor shape")
               (format t "~&Python worker ~A logits max error: ~E~%" device
                       (close-arrays (tb:tensor-array logits) expected
                                     (if (eq device :cpu) 1e-7 3e-5) 1e-5)))
             (signals tb:compatibility-error
               (tb:forward model #2A((3 4 5 6 7) (7 6 5 4 3))))
             ;; A request error must not poison the resident protocol.
             (signals tb:shape-error (tb:forward model #(3 4)))
             (let ((encoded (tb:encode-text model (gethash "text" reference)
                                            :add-special-tokens nil)))
               (check (equalp encoded (gethash "encoded" reference)) "worker tokenizer encode")
               (check (equal (tb:decode-tokens model encoded)
                             (gethash "decoded" reference)) "worker tokenizer decode"))
             (let ((messages (list (tb:make-chat-message "user" "Common Lisp"
                                                         :name "lisp-user")
                                   (tb:make-chat-message "assistant" "Python"))))
               (check (equal (tb:python-apply-chat-template
                              model messages :add-generation-prompt t)
                             (gethash "chat_rendered" reference))
                      "worker chat template rendering")
               (check (equalp (tb:python-apply-chat-template
                               model messages :add-generation-prompt t :tokenize t)
                              (gethash "chat_tokens" reference))
                      "worker chat template tokenization"))
             (check (equalp (tb:generate model #(3 4) :max-new-tokens 4
                                                   :eos-token-id nil)
                            (gethash "generated" reference)) "worker greedy generation")
             (let ((generated
                     (tb:python-generate
                      model '(("input_ids" . #2A((3 4))))
                      :options '(("max_new_tokens" . 4) ("do_sample" . yason:false)
                                 ("num_beams" . 3) ("num_return_sequences" . 2)
                                 ("eos_token_id" . nil) ("pad_token_id" . 0)
                                 ("output_scores" . yason:true))
                      :outputs '("sequences" "sequences_scores"))))
               (unwind-protect
                    (progn
                      (check (equalp
                              (tb:tensor-array
                               (cdr (assoc "sequences" generated :test #'equal)))
                              (json-array (gethash "beam_sequences" reference)))
                             "worker beam sequences")
                      (close-arrays
                       (tb:tensor-array
                        (cdr (assoc "sequences_scores" generated :test #'equal)))
                       (json-array (gethash "beam_scores" reference)) 1e-7 1e-6))
                 (mapc (lambda (entry) (tb:dispose (cdr entry))) generated)))
             (let ((sample nil)
                   (options '(("max_new_tokens" . 4) ("do_sample" . yason:true)
                              ("top_k" . 5) ("temperature" . 0.8)
                              ("eos_token_id" . nil) ("pad_token_id" . 0))))
               (let ((generated (tb:python-processor-generate
                                 model '(("text" . "Common Lisp"))
                                 :options options :seed 1234)))
                 (unwind-protect
                      (setf sample
                            (tb:tensor-array
                             (cdr (assoc "sequences" generated :test #'equal))))
                   (mapc (lambda (entry) (tb:dispose (cdr entry))) generated)))
               (when (eq device :cpu)
                 (check (equalp sample (json-array (gethash "sampled" reference)))
                        "seeded CPU sampling matches independent reference"))
               (let ((repeated (tb:python-processor-generate
                                model '(("text" . "Common Lisp"))
                                :options options :seed 1234)))
                 (unwind-protect
                      (check (equalp
                              sample
                              (tb:tensor-array
                               (cdr (assoc "sequences" repeated :test #'equal))))
                             "seeded sampling repeats on the same device")
                   (mapc (lambda (entry) (tb:dispose (cdr entry))) repeated))))
             (signals tb:compatibility-error (tb:named-parameters model))
             (signals tb:compatibility-error (tb:make-cache model))
             (signals tb:compatibility-error (tb:make-lora model))
             (signals tb:compatibility-error (tb:loss-and-gradients model ids))
             (signals tb:compatibility-error (tb:save-pretrained model fixture))
             (uiop:delete-directory-tree exported :validate t :if-does-not-exist :ignore)
             (let ((sentinel (merge-pathnames "previous-checkpoint" exported)))
               (ensure-directories-exist sentinel)
               (with-open-file (stream sentinel :direction :output :if-exists :supersede)
                 (write-string "worker checkpoint" stream))
               (let ((tb::*before-directory-publication*
                       (lambda (staging published)
                         (declare (ignore staging published))
                         (error "injected worker publication failure"))))
                 (signals error (tb:save-pretrained model exported)))
               (check (equal (uiop:read-file-string sentinel) "worker checkpoint")
                      "failed worker save preserves the previous checkpoint")
               (check (tb:python-worker-alive-p model)
                      "publication failure keeps the worker resident"))
             (tb:save-pretrained model exported)
             (check (not (probe-file (merge-pathnames "previous-checkpoint" exported)))
                    "successful worker save replaces the complete directory")
             (uiop:delete-directory-tree sharded :validate t :if-does-not-exist :ignore)
             (tb:save-pretrained model sharded :max-shard-size 2048)
             (check (probe-file (merge-pathnames "model.safetensors.index.json" sharded))
                    "worker emits the standard shard index")
             (check (tb:python-worker-alive-p model) "save keeps worker resident"))
        (tb:dispose model)
        (tb:dispose model))
      (check (not (tb:python-worker-alive-p model)) "dispose terminates worker")))
  ;; The open AutoClass/tensor interface and training update are separate from
  ;; the compact causal-LM convenience methods above.
  (let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
         (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (fixture (merge-pathnames ".build/fixtures/python-worker-opt/" root))
         (exported (merge-pathnames (format nil ".build/python-worker-trained-~(~A~)/" device) root))
         (reference (tb::read-json (merge-pathnames "worker-reference.json" fixture)))
         (ids (json-array (gethash "input_ids" reference)))
         (mask (json-array (gethash "attention_mask" reference)))
         (labels (json-array (gethash "labels" reference)))
         (texts #("Common Lisp runs Python" "models runs Lisp"))
         (embeds (json-array (gethash "inputs_embeds" reference)))
         (expected-embeds (json-array (gethash "embed_logits" reference)))
         (expected-trained (json-array (gethash "trained_logits" reference)))
         (checkpoint (merge-pathnames
                      (format nil ".build/python-worker-checkpoint-~(~A~)/" device) root))
         (training-dtype (if (eq device :cpu) "bfloat16" "float16"))
         (publication (merge-pathnames
                       (format nil ".build/python-worker-publication-~(~A~)/" device) root))
         (updated-copy nil))
    (tb:with-resource
        (model (tb:from-pretrained fixture :execution :python :device device
                                   :auto-class "AutoModelForCausalLM"
                                   :local-files-only t :max-output-elements 300
                                   :python-threads 1))
      (check (equal (gethash "auto_class" (tb:python-worker-info model))
                    "AutoModelForCausalLM") "dynamic AutoClass selection")
      (let ((processed (tb:python-process model '(("text" . "Common Lisp"))
                                          :outputs '("input_ids" "attention_mask"))))
        (unwind-protect
             (progn
               (check (eq (tb:tensor-dtype
                           (cdr (assoc "input_ids" processed :test #'equal))) :int64)
                      "AutoProcessor preserves integer dtype")
               (check (equalp (tb:tensor-array
                               (cdr (assoc "input_ids" processed :test #'equal)))
                              #2A((3 4))) "AutoProcessor text IDs")
               (check (equalp (tb:tensor-array
                               (cdr (assoc "attention_mask" processed :test #'equal)))
                              #2A((1 1))) "AutoProcessor attention mask"))
          (mapc (lambda (entry) (tb:dispose (cdr entry))) processed)))
      (let ((processed (tb:python-processor-forward
                        model '(("text" . "Common Lisp")) :outputs '("logits"))))
        (unwind-protect
             (tb:with-resource (direct (tb:forward model #2A((3 4))))
               (close-arrays
                (tb:tensor-array (cdr (assoc "logits" processed :test #'equal)))
                (tb:tensor-array direct) 0 0))
          (mapc (lambda (entry) (tb:dispose (cdr entry))) processed)))
      (let ((outputs (tb:python-forward
                      model `(("inputs_embeds" . ,embeds) ("attention_mask" . ,mask))
                      :outputs '("logits"))))
        (unwind-protect
             (close-arrays (tb:tensor-array (cdr (assoc "logits" outputs :test #'equal)))
                           expected-embeds (if (eq device :cpu) 1e-7 3e-5) 1e-5)
          (mapc (lambda (entry) (tb:dispose (cdr entry))) outputs)))
      (tb:with-resource
          (optimizer (tb:make-adamw :learning-rate 0.005 :beta1 0.8 :beta2 0.9
                                    :epsilon 1e-6 :weight-decay 0.01))
        (signals tb:compatibility-error
          (tb:python-train-step model optimizer `(("input_ids" . ,ids))))
        (check (eq (gethash "training" (tb::request-worker model "info")) 'yason:false)
               "failed training restores evaluation mode")
        (loop for expected across (gethash "adamw_losses" reference)
              for step from 0 do
          (let ((actual
                  (case step
                    (0 (tb:python-processor-train-step
                        model optimizer `(("text" . ,texts))
                        :options '(("padding" . yason:true))
                        :model-inputs `(("labels" . ,labels)) :max-grad-norm 1.0))
                    (1 (tb:python-train-step
                        model optimizer
                        `(("input_ids" . ,ids) ("attention_mask" . ,mask)
                          ("labels" . ,labels))
                        :max-grad-norm 1.0))
                    (otherwise
                     (tb:train-step model optimizer ids :labels labels
                                                       :attention-mask mask
                                                       :max-grad-norm 1.0)))))
            (check (< (abs (- actual expected)) (if (eq device :cpu) 1e-7 3e-5))
                   "Lisp-controlled worker AdamW loss")))
        (signals tb:shape-error (tb:python-train-microbatches model optimizer nil))
        (signals tb:compatibility-error
          (tb:configure-python-scheduler model optimizer :linear :total-steps 4)))
      (tb:with-resource (updated (tb:forward model ids :attention-mask mask))
        (setf updated-copy (tb:tensor-array updated))
        (format t "~&Python-trained ~A logits max error: ~E~%" device
                (close-arrays updated-copy expected-trained
                              (if (eq device :cpu) 2e-6 8e-5) 2e-4)))
      (tb:save-pretrained model exported))
    ;; Reload the contributed checkpoint in a new resident process and compare
    ;; against the exact pre-export worker result.
    (tb:with-resource
        (reloaded (tb:from-pretrained exported :execution :python :device device
                                      :auto-class "AutoModelForCausalLM"
                                      :local-files-only t :max-output-elements 300
                                      :python-threads 1))
      (tb:with-resource (logits (tb:forward reloaded ids :attention-mask mask))
        (close-arrays (tb:tensor-array logits) updated-copy 0 0)))
    ;; A standard checkpoint plus process-resident Adam state must resume with
    ;; the same next update in another worker.
    (let ((continued-loss nil) (continued-logits nil))
      (tb:with-resource
          (model (tb:from-pretrained fixture :execution :python :device device
                                     :auto-class "AutoModelForCausalLM"
                                     :local-files-only t :max-output-elements 300
                                     :python-threads 1 :training-dtype training-dtype))
        (check (equal (gethash "training_dtype" (tb:python-worker-info model))
                      training-dtype) "mixed training dtype recorded")
        (tb:with-resource
            (optimizer (tb:make-adamw :learning-rate 0.005 :beta1 0.8 :beta2 0.9
                                      :epsilon 1e-6 :weight-decay 0.01))
          (tb:configure-python-scheduler model optimizer :linear
                                         :warmup-steps 0 :total-steps 3)
          (dotimes (step 2)
            (declare (ignore step))
            (tb:python-train-microbatches
             model optimizer
             `(( ("input_ids" . ,(make-array '(1 4) :displaced-to ids))
                 ("attention_mask" . ,(make-array '(1 4) :displaced-to mask))
                 ("labels" . ,(make-array '(1 4) :displaced-to labels)))
               ( ("input_ids" . ,(make-array '(1 4) :displaced-to ids :displaced-index-offset 4))
                 ("attention_mask" . ,(make-array '(1 4) :displaced-to mask :displaced-index-offset 4))
                 ("labels" . ,(make-array '(1 4) :displaced-to labels :displaced-index-offset 4))))
             :max-grad-norm 1.0))
          (let ((sentinel (merge-pathnames "previous-checkpoint" checkpoint)))
            (uiop:delete-directory-tree checkpoint :validate t :if-does-not-exist :ignore)
            (ensure-directories-exist sentinel)
            (with-open-file (stream sentinel :direction :output :if-exists :supersede)
              (write-string "worker training checkpoint" stream))
            (let ((tb::*before-directory-publication*
                    (lambda (staging published)
                      (declare (ignore published))
                      (when (probe-file (merge-pathnames "tb_training_state.json" staging))
                        (error "injected worker-checkpoint publication failure")))))
              (signals error
                (tb:save-training-checkpoint model optimizer checkpoint)))
            (check (equal (uiop:read-file-string sentinel)
                          "worker training checkpoint")
                   "failed worker checkpoint publication preserves its predecessor")
            (check (not (probe-file (merge-pathnames "tb_training_state.json" checkpoint)))
                   "failed worker checkpoint publication exposes no state sidecar"))
          (tb:save-training-checkpoint model optimizer checkpoint)
          (check (not (probe-file (merge-pathnames "previous-checkpoint" checkpoint)))
                 "worker checkpoint publication replaces the complete directory")
          (setf continued-loss
                (tb:python-train-microbatches
                 model optimizer
                 `(( ("input_ids" . ,ids) ("attention_mask" . ,mask)
                     ("labels" . ,labels)))
                 :max-grad-norm 1.0))
          (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
            (setf continued-logits (tb:tensor-array logits)))))
      (tb:with-resource
          (model (tb:from-pretrained checkpoint :execution :python :device device
                                     :auto-class "AutoModelForCausalLM"
                                     :local-files-only t :max-output-elements 300
                                     :python-threads 1 :training-dtype training-dtype))
        (tb:with-resource (wrong (tb:make-sgd :learning-rate 0.005))
          (signals tb:compatibility-error
            (tb:restore-training-checkpoint model wrong checkpoint)))
        (tb:with-resource
            (optimizer (tb:make-adamw :learning-rate 0.005 :beta1 0.8 :beta2 0.9
                                      :epsilon 1e-6 :weight-decay 0.01))
          (check (= 2 (tb:restore-training-checkpoint model optimizer checkpoint))
                 "optimizer step restored")
          (let ((information (tb:python-optimizer-info model optimizer)))
            (check (equal (gethash "scheduler" information) "linear")
                   "scheduler restored")
            (check (< (abs (- (gethash "learning_rate" information)
                              (/ 0.005 3))) 1e-9)
                   "scheduler learning rate restored"))
          (let ((resumed-loss
                  (tb:python-train-microbatches
                   model optimizer
                   `(( ("input_ids" . ,ids) ("attention_mask" . ,mask)
                       ("labels" . ,labels)))
                   :max-grad-norm 1.0)))
            (check (< (abs (- resumed-loss continued-loss))
                      (if (eq device :cpu) 1e-7 3e-5))
                   "resumed AdamW loss"))
          (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
            (close-arrays (tb:tensor-array logits) continued-logits
                          (if (eq device :cpu) 1e-7 3e-5) 1e-5)))
        (signals tb:shape-error (tb:push-to-hub model "invalid repository name"))
        (let ((result (tb:push-to-hub
                       model "common-lisp/test-model" :private t
                       :commit-message "Test trained export" :model-card "# Test model"
                       :dry-run-directory publication)))
          (check (equal (gethash "repo_id" result) "common-lisp/test-model")
                 "Hub publication repository")
          (check (probe-file (merge-pathnames "README.md" publication))
                 "Hub publication includes model card")
          (check (probe-file (merge-pathnames "model.safetensors" publication))
                 "Hub publication includes safe weights")))))
  ;; Standard PEFT adapters can enter from Python, be manipulated and trained
  ;; in the resident worker from Lisp, and leave as either PEFT or merged
  ;; Transformers artifacts.
  (let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
         (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (fixture (merge-pathnames ".build/fixtures/python-worker-opt/" root))
         (reference (tb::read-json (merge-pathnames "worker-reference.json" fixture)))
         (ids (json-array (gethash "input_ids" reference)))
         (mask (json-array (gethash "attention_mask" reference)))
         (labels (json-array (gethash "labels" reference)))
         (expected (json-array (gethash "peft_logits" reference)))
         (python-adapter (merge-pathnames "python-lora/" fixture))
         (loaded-export (merge-pathnames
                         (format nil ".build/python-worker-loaded-adapter-~(~A~)/" device) root))
         (loaded-merged (merge-pathnames
                         (format nil ".build/python-worker-loaded-merged-~(~A~)/" device) root))
         (trained-export (merge-pathnames
                          (format nil ".build/python-worker-trained-adapter-~(~A~)/" device) root))
         (trained-merged (merge-pathnames
                          (format nil ".build/python-worker-trained-merged-~(~A~)/" device) root))
         (peft-checkpoint (merge-pathnames
                           (format nil ".build/python-worker-peft-checkpoint-~(~A~)/" device) root))
         (publication (merge-pathnames
                       (format nil ".build/python-worker-adapter-publication-~(~A~)/" device) root)))
    (tb:with-resource
        (model (tb:from-pretrained fixture :execution :python :device device
                                   :auto-class "AutoModelForCausalLM"
                                   :local-files-only t :python-threads 1))
      (let ((information (tb:python-load-adapter
                          model python-adapter :trainable t :local-files-only t)))
        (check (equal (gethash "peft_type" information) "LORA")
               "load Python-created PEFT adapter")
        (check (equalp (gethash "active_adapters" information) #("default"))
               "loaded adapter is active")
        (check (< 0 (gethash "trainable_parameters" information)
                    (gethash "total_parameters" information))
               "PEFT freezes the base model"))
      (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
        (close-arrays (tb:tensor-array logits) expected
                      (if (eq device :cpu) 2e-7 4e-5) 2e-5))
      (let ((sentinel (merge-pathnames "previous-adapter" loaded-export)))
        (uiop:delete-directory-tree loaded-export :validate t :if-does-not-exist :ignore)
        (ensure-directories-exist sentinel)
        (with-open-file (stream sentinel :direction :output :if-exists :supersede)
          (write-string "worker adapter" stream))
        (let ((tb::*before-directory-publication*
                (lambda (staging published)
                  (declare (ignore staging published))
                  (error "injected worker-adapter publication failure"))))
          (signals error (tb:python-save-adapter model loaded-export)))
        (check (equal (uiop:read-file-string sentinel) "worker adapter")
               "failed worker adapter publication preserves its predecessor"))
      (tb:python-save-adapter model loaded-export)
      (check (not (probe-file (merge-pathnames "previous-adapter" loaded-export)))
             "worker adapter publication replaces the complete directory")
      (let ((published (tb:push-to-hub
                        model "common-lisp/test-adapter" :model-card "# Test adapter"
                        :dry-run-directory publication)))
        (check (find "adapter_config.json" (gethash "files" published) :test #'equal)
               "adapter publication stages PEFT configuration")
        (check (find "adapter_model.safetensors" (gethash "files" published) :test #'equal)
               "adapter publication stages safe PEFT weights"))
      (tb:python-merge-adapter model :safe-merge t)
      (check (null (gethash "peft_type" (tb:python-adapter-info model)))
             "merge removes PEFT wrapper")
      (tb:save-pretrained model loaded-merged))
    (let ((trained-logits nil) (continued-loss nil))
      (tb:with-resource
          (model (tb:from-pretrained fixture :execution :python :device device
                                     :auto-class "AutoModelForCausalLM"
                                     :local-files-only t :python-threads 1))
        (let ((information
                (tb:python-make-lora model :rank 2 :alpha 4
                                          :target-modules '("q_proj" "v_proj")
                                          :rslora t :task-type "CAUSAL_LM")))
          (check (equal (gethash "peft_type" information) "LORA")
                 "create standard worker LoRA")
          (check (eq (gethash "use_rslora" information) 'yason:true)
                 "rank-stabilized LoRA configuration"))
        (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.01))
          (let ((loss (tb:python-train-step
                       model optimizer
                       `(("input_ids" . ,ids) ("attention_mask" . ,mask)
                         ("labels" . ,labels)) :max-grad-norm 1.0)))
            (check (and (realp loss) (plusp loss)) "train worker LoRA"))
          (let ((sentinel (merge-pathnames "previous-checkpoint" peft-checkpoint)))
            (uiop:delete-directory-tree peft-checkpoint :validate t
                                                        :if-does-not-exist :ignore)
            (ensure-directories-exist sentinel)
            (with-open-file (stream sentinel :direction :output :if-exists :supersede)
              (write-string "worker PEFT checkpoint" stream))
            (let ((tb::*before-directory-publication*
                    (lambda (staging published)
                      (declare (ignore published))
                      (when (probe-file (merge-pathnames "tb_training_state.json" staging))
                        (error "injected worker-PEFT checkpoint publication failure")))))
              (signals error
                (tb:save-training-checkpoint model optimizer peft-checkpoint)))
            (check (equal (uiop:read-file-string sentinel) "worker PEFT checkpoint")
                   "failed worker PEFT checkpoint preserves its predecessor"))
          (tb:save-training-checkpoint model optimizer peft-checkpoint)
          (check (not (probe-file (merge-pathnames "previous-checkpoint" peft-checkpoint)))
                 "worker PEFT checkpoint replaces the complete directory")
          (setf continued-loss
                (tb:python-train-step
                 model optimizer
                 `(("input_ids" . ,ids) ("attention_mask" . ,mask)
                   ("labels" . ,labels)) :max-grad-norm 1.0))
          (signals tb:compatibility-error (tb:python-merge-adapter model)))
        (tb:python-save-adapter model trained-export)
        (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
          (setf trained-logits (tb:tensor-array logits)))
        (tb:python-merge-adapter model)
        (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
          (close-arrays (tb:tensor-array logits) trained-logits
                        (if (eq device :cpu) 2e-6 5e-5) 2e-5))
        (tb:save-pretrained model trained-merged))
      (tb:with-resource
          (model (tb:from-pretrained fixture :execution :python :device device
                                     :auto-class "AutoModelForCausalLM"
                                     :local-files-only t :python-threads 1))
        (tb:python-load-adapter model peft-checkpoint :trainable t :local-files-only t)
        (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.01))
          (check (= 1 (tb:restore-training-checkpoint model optimizer peft-checkpoint))
                 "restore PEFT optimizer step")
          (let ((resumed
                  (tb:python-train-step
                   model optimizer
                   `(("input_ids" . ,ids) ("attention_mask" . ,mask)
                     ("labels" . ,labels)) :max-grad-norm 1.0)))
            (check (< (abs (- resumed continued-loss))
                      (if (eq device :cpu) 2e-7 4e-5))
                   "resume PEFT optimizer exactly"))
          (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
            (close-arrays (tb:tensor-array logits) trained-logits
                          (if (eq device :cpu) 2e-6 5e-5) 2e-5))))
      (tb:with-resource
          (model (tb:from-pretrained fixture :execution :python :device device
                                     :auto-class "AutoModelForCausalLM"
                                     :local-files-only t :python-threads 1))
        (tb:python-load-adapter model trained-export :local-files-only t)
        (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
          (close-arrays (tb:tensor-array logits) trained-logits
                        (if (eq device :cpu) 2e-6 5e-5) 2e-5)))))
  ;; The opt-in binary path must carry both named tensor inputs and outputs
  ;; without embedding their elements in JSON protocol frames.
  (let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
         (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (fixture (merge-pathnames ".build/fixtures/python-worker-opt/" root))
         (embeds (make-array '(4 16 16) :element-type 'single-float))
         (mask (make-array '(4 16) :initial-element 1)))
    (dotimes (index (array-total-size embeds))
      (setf (row-major-aref embeds index)
            (coerce (/ (- (mod index 31) 15) 100.0) 'single-float)))
    (tb:with-resource
        (model (tb:from-pretrained fixture :execution :python :device device
                                   :auto-class "AutoModelForCausalLM"
                                   :local-files-only t :max-output-elements 3000
                                   :python-threads 1))
      (let ((json (tb:python-forward
                   model `(("inputs_embeds" . ,embeds) ("attention_mask" . ,mask))))
            (binary (tb:python-forward
                     model `(("inputs_embeds" . ,embeds) ("attention_mask" . ,mask))
                     :transport :binary)))
        (unwind-protect
             (close-arrays
              (tb:tensor-array (cdr (assoc "logits" binary :test #'equal)))
              (tb:tensor-array (cdr (assoc "logits" json :test #'equal))) 0 0)
          (mapc (lambda (entry) (tb:dispose (cdr entry))) json)
          (mapc (lambda (entry) (tb:dispose (cdr entry))) binary)))
      (let ((information (tb::request-worker model "info")))
        (check (= 2 (gethash "binary_inputs" information))
               "binary worker consumed both tensor inputs")
        (check (= 1 (gethash "binary_outputs" information))
               "binary worker produced one tensor output"))))
  (format t "Python worker acceptance tests passed.~%"))
