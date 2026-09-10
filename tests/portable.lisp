(in-package #:tb-tests)

(defun run-portable-tests ()
  (let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
         (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (fixture (merge-pathnames ".build/fixtures/portable/" root))
         (reference (tb::read-json (merge-pathnames "reference.json" fixture)))
         (ids (json-array (gethash "input_ids" reference)))
         (mask (json-array (gethash "attention_mask" reference)))
         (labels (json-array (gethash "labels" reference)))
         (expected (json-array (gethash "logits" reference)))
         (exported (merge-pathnames
                    (format nil ".build/portable-export-~(~A~)/" device) root))
         (sharded (merge-pathnames
                   (format nil ".build/portable-sharded-~(~A~)/" device) root))
         (updated (merge-pathnames
                   (format nil ".build/portable-updated-~(~A~)/" device) root))
         (lisp-created (merge-pathnames
                        (format nil ".build/portable-lisp-created-~(~A~)/" device)
                        root))
         (publication (merge-pathnames
                       (format nil ".build/portable-publication-~(~A~)/" device)
                       root)))
    (let ((config (tb:make-portable-config
                   :vocab-size 32 :hidden-size 16 :intermediate-size 40
                   :layers 2 :attention-heads 4 :kv-heads 2
                   :context-length 64)))
      (check (equal (gethash "model_type" config) "tb_parallel")
             "portable config has an honest custom model type")
      (check (= (gethash "portable_architecture_version" config) 1)
             "portable config format is versioned")
      (check (tb::json-true-p (gethash "parallel_residual" config))
             "portable config records its defining residual topology"))
    (tb:with-resource
        (first (tb:make-portable-causal-lm
                :vocab-size 32 :hidden-size 16 :intermediate-size 40
                :layers 2 :attention-heads 4 :kv-heads 2 :context-length 64
                :tie-word-embeddings t :seed 104729 :device device))
      (tb:with-resource
          (second (tb:make-portable-causal-lm
                   :vocab-size 32 :hidden-size 16 :intermediate-size 40
                   :layers 2 :attention-heads 4 :kv-heads 2 :context-length 64
                   :tie-word-embeddings t :seed 104729 :device device))
        (check (= (length (tb:named-parameters first)) 18)
               "Lisp constructor creates the complete tied parameter tree")
        (check (null (getf (tb:model-capabilities first) :tokenizer))
               "Lisp constructor does not infer tokenizer assets from the working directory")
        (dolist (entry (tb:named-parameters first))
          (check (equalp (tb:tensor-array (cdr entry))
                         (tb:tensor-array
                          (cdr (assoc (car entry) (tb:named-parameters second)
                                      :test #'equal))))
                 "portable seed initialization is reproducible")))
      (tb:with-resource (logits (tb:forward first ids :attention-mask mask))
        (check (eq (tb::require-finite logits) logits)
               "Lisp-created portable logits are finite"))
      (signals tb:compatibility-error (tb:encode-text first "Common Lisp"))
      (signals tb:compatibility-error (tb:encode-batch first '("Common Lisp")))
      (signals tb:compatibility-error (tb:attach-tokenizer first fixture))
      (check (eq first (tb:attach-tokenizer first (merge-pathnames "text-assets/" fixture)))
             "tokenizer attachment returns the model")
      (check (equalp (tb:encode-text first "Common Lisp" :add-special-tokens nil) #(3 4))
             "attached tokenizer encodes known text")
      (check (equal (tb:decode-tokens first #(3 4)) "Common Lisp")
             "attached tokenizer decodes known IDs")
      (check (getf (tb:model-capabilities first) :tokenizer)
             "attached tokenizer is discoverable")
      (multiple-value-bind (batch mask types)
          (tb:encode-batch first '("Common Lisp" "Lisp") :pad-token-id 0)
        (check (equalp batch #2A((3 4) (4 0))) "attached tokenizer batches text")
        (check (equalp mask #2A((1 1) (1 0))) "attached tokenizer supplies right padding")
        (check (equalp types #2A((0 0) (0 0))) "attached tokenizer supplies type IDs")
        (tb:with-resource (logits (tb:forward first batch :attention-mask mask))
          (check (eq logits (tb::require-finite logits)) "batch text feeds Lisp-created model")))
      (signals tb:compatibility-error
        (tb:attach-tokenizer first (merge-pathnames "invalid-text-assets/" fixture)))
      (check (equalp (tb:encode-text first "Common Lisp") #(3 4))
             "failed attachment preserves the previous tokenizer")
      ;; Snapshot assets as bytes: later edits to the caller's files must not
      ;; make exported text semantics differ from the resident tokenizer.
      (let ((mutable (merge-pathnames
                      (format nil ".build/mutable-tokenizer-~(~A~)/" device) root)))
        (ensure-directories-exist mutable)
        (dolist (name '("tokenizer.json" "tokenizer_config.json"))
          (uiop:copy-file (merge-pathnames name (merge-pathnames "text-assets/" fixture))
                          (merge-pathnames name mutable)))
        (tb:attach-tokenizer first mutable)
        (with-open-file (stream (merge-pathnames "tokenizer.json" mutable)
                                :direction :output :if-exists :supersede)
          (write-string "{}" stream))
        (signals tb:compatibility-error (tb:attach-tokenizer first mutable)))
      (check (equalp (tb:generate first "Common Lisp" :max-new-tokens 2 :eos-token-id nil)
                     (tb:generate first #(3 4) :max-new-tokens 2 :eos-token-id nil))
             "text and ID generation agree")
      (tb:save-pretrained first lisp-created)
      (check (equal (uiop:read-file-string (merge-pathnames "tokenizer.json" lisp-created))
                    (uiop:read-file-string
                     (merge-pathnames "text-assets/tokenizer.json" fixture)))
             "export contains the original tokenizer byte snapshot")
      (tb:with-resource (reloaded (tb:from-pretrained lisp-created :device device))
        (check (equalp (tb:encode-text reloaded "Common Lisp") #(3 4))
               "native reload preserves attached tokenizer"))
      (tb:with-resource (optimizer (tb:make-adamw :learning-rate 0.001))
        (let ((checkpoint (merge-pathnames
                           (format nil ".build/portable-text-checkpoint-~(~A~)/" device)
                           root)))
          (tb:train-step first optimizer ids :labels labels :attention-mask mask)
          (tb:save-training-checkpoint first optimizer checkpoint)
          (tb:with-resource (resumed (tb:from-pretrained checkpoint :device device))
            (check (equalp (tb:encode-text resumed "Common Lisp") #(3 4))
                   "training checkpoint includes attached text assets")
            (tb:with-resource (resumed-optimizer (tb:make-adamw :learning-rate 0.001))
              (tb:restore-training-checkpoint resumed resumed-optimizer checkpoint)
              (check (= (tb:train-step first optimizer ids :labels labels :attention-mask mask)
                        (tb:train-step resumed resumed-optimizer ids :labels labels :attention-mask mask))
                     "Lisp-created model resumes the exact next training loss")))))
      (let ((result (tb:push-to-hub
                     first "common-lisp/portable-test"
                     :dry-run-directory publication
                     :model-card "# Portable Lisp model")))
        (check (find "configuration_tb_parallel.py" (gethash "files" result)
                     :test #'equal)
               "Hub stage includes the custom configuration implementation")
        (check (find "modeling_tb_parallel.py" (gethash "files" result)
                     :test #'equal)
               "Hub stage includes the custom model implementation")
        (check (find "tokenizer.json" (gethash "files" result) :test #'equal)
               "Hub stage includes the attached tokenizer")))
    (let* ((config (alexandria:copy-hash-table
                    (tb::read-json (merge-pathnames "config.json" fixture))))
           (mapping (make-hash-table :test 'equal)))
      (setf (gethash "AutoConfig" mapping) "configuration_other.OtherConfig"
            (gethash "auto_map" config) mapping)
      (signals tb:compatibility-error (tb::validate-config config)))
    (let* ((capabilities (tb:inspect-pretrained fixture))
           (operations (getf capabilities :operations)))
      (check (getf capabilities :supported-p) "portable model is natively supported")
      (check (eq (getf capabilities :architecture) :tb_parallel)
             "portable architecture is discoverable")
      (check (every (lambda (operation) (member operation operations))
                    '(:forward :training :training-checkpoint :export :cache :generation))
             "portable native operations are discoverable")
      (check (member :custom-python-export operations)
             "portable capability reports emitted Python execution code"))
    (tb:with-resource (model (tb:from-pretrained fixture :device device))
      (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
        (format t "~&Portable ~A logits max error: ~E~%" device
                (close-arrays (tb:tensor-array logits) expected 3e-5 3e-4)))
      (tb:with-resource (cache (tb:make-cache model :growth-step 4))
        (dotimes (position 4)
          (let ((step (make-array '(2 1)))
                (padding (make-array (list 2 (1+ position)))))
            (dotimes (batch 2)
              (setf (aref step batch 0) (aref ids batch position))
              (dotimes (i (1+ position))
                (setf (aref padding batch i) (aref mask batch i))))
            (tb:with-resource (logits (tb:forward model step :cache cache
                                                        :attention-mask padding))
              (let* ((actual (tb:tensor-array logits))
                     (slice (make-array (array-dimensions actual))))
                (dotimes (batch 2)
                  (dotimes (token (array-dimension expected 2))
                    (setf (aref slice batch 0 token)
                          (aref expected batch position token))))
                (close-arrays actual slice 3e-5 3e-4))))))
      (tb:save-pretrained model exported)
      (dolist (file '("configuration_tb_parallel.py" "modeling_tb_parallel.py"))
        (check (probe-file (merge-pathnames file exported))
               (format nil "portable export includes ~A" file)))
      (let* ((config (tb::read-json (merge-pathnames "config.json" exported)))
             (mapping (gethash "auto_map" config)))
        (check (equal (gethash "AutoConfig" mapping)
                      "configuration_tb_parallel.TBParallelConfig")
               "portable export registers AutoConfig")
        (check (equal (gethash "AutoModelForCausalLM" mapping)
                      "modeling_tb_parallel.TBParallelForCausalLM")
               "portable export registers AutoModelForCausalLM"))
      (signals tb:compatibility-error
        (tb:from-pretrained exported :execution :python :device device
                                     :local-files-only t
                                     :auto-class "AutoModelForCausalLM"))
      (tb:with-resource
          (python-model
           (tb:from-pretrained exported :execution :python :device device
                                        :local-files-only t :trust-remote-code t
                                        :auto-class "AutoModelForCausalLM"
                                        :max-output-elements 300
                                        :python-threads 1))
        (tb:with-resource (logits (tb:forward python-model ids :attention-mask mask))
          (close-arrays (tb:tensor-array logits) expected
                        (if (eq device :cpu) 1e-7 3e-5) 3e-4))
        (check (equalp (tb:generate python-model #(3 4) :max-new-tokens 4
                                                       :eos-token-id nil)
                       (tb:generate model #(3 4) :max-new-tokens 4
                                                  :eos-token-id nil))
               "emitted Python and native greedy generation agree"))
      (tb:save-pretrained model sharded :max-shard-size 2048)
      (check (probe-file (merge-pathnames "model.safetensors.index.json" sharded))
             "portable architecture supports standard sharded export")
      (multiple-value-bind (loss gradients)
          (tb:loss-and-gradients model ids :labels labels :attention-mask mask)
        (let ((expected-gradients (make-hash-table :test 'equal)))
          (unwind-protect
               (progn
                 (tb::load-weights (merge-pathnames "gradients.safetensors" fixture)
                                   (tb:model-backend model) expected-gradients)
                 (check (< (abs (- loss (gethash "loss" reference))) 3e-5)
                        "portable shifted loss matches independent Torch")
                 (check (= (length gradients) (hash-table-count expected-gradients))
                        "portable gradient coverage is exact")
                 (dolist (entry gradients)
                   (close-arrays
                    (tb:tensor-array (cdr entry))
                    (tb:tensor-array (gethash (car entry) expected-gradients))
                    4e-6 4e-4)))
            (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients)
            (maphash (lambda (name value)
                       (declare (ignore name))
                       (tb:dispose value))
                     expected-gradients))))
      (tb:sgd-step model ids :labels labels :attention-mask mask :learning-rate 0.01)
      (tb:save-pretrained model updated))
    (format t "Portable custom-architecture acceptance tests passed.~%")))

(defun run-portable-python-resave-test ()
  (let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
         (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (directory
           (merge-pathnames
            (format nil ".build/portable-lisp-created-~(~A~)-python-resaved/" device)
            root))
         (reference (tb::read-json (merge-pathnames "python-reference.json" directory)))
         (ids (json-array (gethash "input_ids" reference)))
         (mask (json-array (gethash "attention_mask" reference)))
         (expected (json-array (gethash "logits" reference))))
    (tb:with-resource (model (tb:from-pretrained directory :device device))
      (check (equalp (tb:encode-text model "Common Lisp") #(3 4))
             "Python-resaved tokenizer returns to native Lisp")
      (tb:with-resource (logits (tb:forward model ids :attention-mask mask))
        (format t "~&Python-trained portable ~A logits max error: ~E~%"
                device
                (close-arrays (tb:tensor-array logits) expected 3e-5 3e-4))))
    (format t "Python-resaved portable model returned to native Lisp.~%")))
