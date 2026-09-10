(in-package #:tb)

(defun read-json (path)
  (with-open-file (stream path :external-format :utf-8)
    (yason:parse stream :json-arrays-as-vectors t :json-booleans-as-symbols t)))
(defun write-json (value path)
  (with-open-file (stream path :direction :output :if-exists :supersede :external-format :utf-8)
    (yason:encode value stream) (terpri stream)))
(defun json-true-p (value) (eq value 'yason:true))
(defclass model ()
  ((config :initarg :config :reader model-config)
   (backend :initarg :backend :reader model-backend)
   (directory :initarg :directory :reader model-directory)
   (parameters :initarg :parameters :accessor parameters)
   (tokenizer :initform nil :accessor model-tokenizer)
   (tokenizer-assets :initform nil :accessor model-tokenizer-assets)
   (version :initform 0 :accessor model-version)
   (adapter :initform nil :accessor model-adapter)
   (base-updated :initform nil :accessor base-updated-p)
   (source-id :initarg :source-id :reader model-source-id)
   (source-revision :initarg :source-revision :reader model-source-revision)))
(defclass llama-model (model) ())
(defvar *architectures* (make-hash-table :test 'equal))
(defun register-architecture (model-type class)
  "Register a CLOS model class whose schema, configuration, and graph methods are defined."
  (unless (and (stringp model-type) (subtypep class 'model))
    (fail 'compatibility-error "An architecture must name a MODEL subclass"))
  (setf (gethash model-type *architectures*) class))
(register-architecture "llama" 'llama-model)
(defun architecture-class (c)
  (unless (hash-table-p c) (fail 'compatibility-error "Model configuration must be an object"))
  (or (gethash (gethash "model_type" c) *architectures*)
      (fail 'compatibility-error "Unregistered model architecture: ~S" (gethash "model_type" c))))
(defgeneric sanitize-architecture-weights (architecture config weights))
(defmethod sanitize-architecture-weights (architecture config weights)
  (declare (ignore architecture config weights)) nil)
(defgeneric validate-architecture-config (architecture config))
(defgeneric architecture-parameter-schema (architecture config))
(defgeneric embedding-weight-name (architecture))
(defgeneric default-tie-embeddings (architecture))
(defmethod embedding-weight-name ((architecture (eql 'llama-model))) "model.embed_tokens.weight")
(defmethod default-tie-embeddings ((architecture (eql 'llama-model))) 'yason:false)
(defmethod architecture-capabilities (architecture config directory)
  (declare (ignore architecture config directory))
  (list :operations '(:forward :training :training-checkpoint :export :cache :generation :lora :rslora)
        :mask-policy :right-padding
        :restrictions
        '("FP32 execution and export"
          "unscaled non-interleaved RoPE"
          "greedy single-sequence generation"
          "zero-dropout training")))
(defun tied-embeddings-p (c)
  (json-true-p (gethash "tie_word_embeddings" c (default-tie-embeddings (architecture-class c)))))
(defun validate-config (c) (validate-architecture-config (architecture-class c) c))
(defun parameter-schema (c) (architecture-parameter-schema (architecture-class c) c))
(defgeneric model-layer-count (model))
(defgeneric model-context-length (model))
(defmethod model-layer-count ((model llama-model)) (gethash "num_hidden_layers" (model-config model)))
(defmethod model-context-length ((model llama-model)) (gethash "max_position_embeddings" (model-config model)))
(defgeneric validate-training-config (model))
(defmethod validate-training-config ((model model)) t)
(defmethod named-parameters ((model model))
  "Sorted association list of canonical Hugging Face names and borrowed tensors."
  (sort (loop for k being the hash-keys of (parameters model) using (hash-value v)
              collect (cons k v)) #'string< :key #'car))
(defmethod python-architecture-files ((model model))
  (declare (ignore model))
  nil)
(defmethod dispose ((model model))
  (maphash (lambda (k v) (declare (ignore k)) (dispose v)) (parameters model))
  (clrhash (parameters model))
  (dispose (model-adapter model)) (setf (model-adapter model) nil)
  (dispose (model-tokenizer model)) (setf (model-tokenizer model) nil)
  (setf (model-tokenizer-assets model) nil)
  (dispose (model-backend model)))
(defun native-tokenizer-capability (directory)
  (and directory (probe-file (merge-pathnames "tokenizer.json" directory))
       :tokenizer-json))
(defun native-capability-report (config directory source revision &key model)
  (let* ((architecture (architecture-class config))
         (specific (architecture-capabilities architecture config directory))
         (operations (copy-list (getf specific :operations)))
         (tokenizer (if (and model (model-tokenizer-assets model))
                        :tokenizer-json (native-tokenizer-capability directory))))
    (when tokenizer
      (pushnew :tokenization operations)
      (pushnew :batch-tokenization operations))
    (list :supported-p t
          :loaded-p (not (null model))
          :execution :native
          :backend :mlx
          :model-type (gethash "model_type" config)
          :architecture (intern (string-upcase (gethash "model_type" config)) :keyword)
          :source source
          :revision revision
          :supported-devices '(:cpu :gpu)
          :device (and model (backend-device (model-backend model)))
          :requested-device (and model (backend-device (model-backend model)))
          :device-api (and model (eq (backend-device (model-backend model)) :gpu)
                           #+darwin :metal #-darwin :cuda)
          :operations (sort operations #'string< :key #'symbol-name)
          :checkpoint-dtype (or (gethash "dtype" config) (gethash "torch_dtype" config)
                                "float32")
          :execution-dtype :float32
          :precision-policy :convert-to-float32
          :tokenizer tokenizer
          :quantization nil
          :compiled-p nil
          :mask-policy (getf specific :mask-policy)
          :restrictions (getf specific :restrictions)
          :parameter-count (and model (length (named-parameters model)))
          :python-fallback-p t
          :reason nil)))
(defun unsupported-capability-report (config directory source revision condition)
  (let ((model-type (and (hash-table-p config) (gethash "model_type" config))))
    (list :supported-p nil
          :loaded-p nil
          :execution :native
          :backend :mlx
          :model-type model-type
          :architecture (and model-type (gethash model-type *architectures*)
                             (intern (string-upcase model-type) :keyword))
          :source source
          :revision revision
          :supported-devices '(:cpu :gpu)
          :device nil
          :requested-device nil
          :device-api nil
          :operations nil
          :checkpoint-dtype (and (hash-table-p config)
                                 (or (gethash "dtype" config) (gethash "torch_dtype" config)
                                     "float32"))
          :execution-dtype :float32
          :precision-policy :convert-to-float32
          :tokenizer (native-tokenizer-capability directory)
          :quantization (and (hash-table-p config) (gethash "quantization_config" config))
          :compiled-p nil
          :mask-policy nil
          :restrictions nil
          :parameter-count nil
          :python-fallback-p t
          :reason (princ-to-string condition))))
(defun inspect-pretrained (source &key revision (local-files-only nil) cache-directory
                                       python-executable)
  "Inspect native configuration compatibility without loading weights or allocating an MLX device.
Unknown or incompatible configurations return a report with :SUPPORTED-P NIL and an explicit
:REASON; Hub sources use the same pinned safe-artifact resolver as FROM-PRETRAINED."
  (let* ((local (probe-file source))
         (source-id (if local (namestring (truename source)) source))
         (resolved-revision revision)
         (directory
           (if local
               (uiop:ensure-directory-pathname local)
               (multiple-value-bind (path resolved)
                   (download-from-hub source :revision revision
                                             :local-files-only local-files-only
                                             :cache-directory cache-directory
                                             :python-executable python-executable)
                 (setf resolved-revision resolved)
                 path)))
         (config-path (merge-pathnames "config.json" directory))
         (config nil))
    (handler-case
        (progn
          (unless (probe-file config-path)
            (fail 'compatibility-error "Checkpoint has no config.json: ~A" directory))
          (setf config (read-json config-path))
          (validate-config config)
          (native-capability-report config directory source-id resolved-revision))
      (compatibility-error (condition)
        (unsupported-capability-report config directory source-id resolved-revision condition)))))
(defmethod model-capabilities ((model model))
  (native-capability-report (model-config model) (model-directory model)
                            (model-source-id model) (model-source-revision model)
                            :model model))
(defun config (model name &optional default) (gethash name (model-config model) default))
(defun head-dimension (config)
  (gethash "head_dim" config (/ (gethash "hidden_size" config) (gethash "num_attention_heads" config))))
(defun rope-base (config)
  (let ((parameters (gethash "rope_parameters" config)))
    (if parameters (gethash "rope_theta" parameters 10000.0)
        (gethash "rope_theta" config 10000.0))))
(defparameter +llama-config-keys+
  '("architectures" "model_type" "hidden_size" "intermediate_size" "vocab_size"
    "num_hidden_layers" "num_attention_heads" "num_key_value_heads" "head_dim"
    "max_position_embeddings" "rms_norm_eps" "hidden_act" "attention_bias" "mlp_bias"
    "attention_dropout" "pretraining_tp" "rope_theta" "rope_scaling" "rope_parameters"
    "rope_interleaved" "tie_word_embeddings" "initializer_range" "is_llama_config"
    "bos_token_id" "eos_token_id" "pad_token_id" "use_cache" "dtype" "torch_dtype"
    "transformers_version" "_name_or_path" "_commit_hash" "_attn_implementation_autoset"
    "return_dict" "output_attentions" "output_hidden_states"))
(defmethod validate-architecture-config ((architecture (eql 'llama-model)) c)
  (unless (and (hash-table-p c) (equal (gethash "model_type" c) "llama"))
    (fail 'compatibility-error "Only the registered Llama architecture is supported"))
  (maphash (lambda (key value) (declare (ignore value))
             (unless (member key +llama-config-keys+ :test #'equal)
               (fail 'compatibility-error "Unrecognized Llama configuration field: ~A" key))) c)
  (when (json-true-p (gethash "rope_interleaved" c))
    (fail 'compatibility-error "Interleaved RoPE is unsupported"))
  (dolist (key '("vocab_size" "hidden_size" "intermediate_size" "num_hidden_layers"
                 "num_attention_heads" "max_position_embeddings"))
    (unless (typep (gethash key c) '(integer 1 2147483647))
      (fail 'compatibility-error "Invalid positive dimension: ~A" key)))
  (let* ((heads (gethash "num_attention_heads" c))
         (kv (gethash "num_key_value_heads" c heads)) (dim (head-dimension c)))
    (unless (and (typep kv '(integer 1)) (zerop (mod heads kv))
                 (typep dim '(integer 2)) (evenp dim))
      (fail 'compatibility-error "Invalid attention/grouped-head dimensions")))
  (unless (equal (gethash "hidden_act" c "silu") "silu")
    (fail 'compatibility-error "Only SiLU activation is supported"))
  (dolist (key '("attention_bias" "mlp_bias" "quantization_config" "sliding_window" "auto_map"))
    (let ((value (gethash key c)))
      (when (and value (not (eq value 'yason:false)))
        (fail 'compatibility-error "Unsupported feature: ~A" key))))
  (unless (and (zerop (gethash "attention_dropout" c 0))
               (= 1 (gethash "pretraining_tp" c 1)))
    (fail 'compatibility-error "Dropout and pretraining tensor parallel variants are unsupported"))
  (let ((scaling (gethash "rope_scaling" c)) (rp (gethash "rope_parameters" c)))
    (when rp
      (unless (hash-table-p rp) (fail 'compatibility-error "Invalid rope_parameters"))
      (maphash (lambda (key value) (declare (ignore value))
                 (unless (member key '("rope_type" "rope_theta") :test #'equal)
                   (fail 'compatibility-error "Unsupported RoPE parameter: ~A" key))) rp))
    (when (or scaling (and rp (not (equal (gethash "rope_type" rp) "default"))))
      (fail 'compatibility-error "Only unscaled default RoPE is supported")))
  (unless (and (realp (rope-base c)) (> (rope-base c) 0)
               (realp (gethash "rms_norm_eps" c)) (> (gethash "rms_norm_eps" c) 0))
    (fail 'compatibility-error "Invalid RoPE base or normalization epsilon"))
  c)
(defmethod architecture-parameter-schema ((architecture (eql 'llama-model)) c)
  "One canonical schema drives validation and parameter introspection. Weights are (out,in)."
  (let* ((d (gethash "hidden_size" c)) (f (gethash "intermediate_size" c))
         (v (gethash "vocab_size" c)) (h (gethash "num_attention_heads" c))
         (k (gethash "num_key_value_heads" c h)) (hd (head-dimension c))
         (schema (list (cons "model.embed_tokens.weight" (list v d))
                       (cons "model.norm.weight" (list d)))))
    (unless (tied-embeddings-p c)
      (push (cons "lm_head.weight" (list v d)) schema))
    (dotimes (i (gethash "num_hidden_layers" c))
      (dolist (entry `(("self_attn.q_proj.weight" ,(* h hd) ,d)
                       ("self_attn.k_proj.weight" ,(* k hd) ,d)
                       ("self_attn.v_proj.weight" ,(* k hd) ,d)
                       ("self_attn.o_proj.weight" ,d ,(* h hd))
                       ("mlp.gate_proj.weight" ,f ,d) ("mlp.up_proj.weight" ,f ,d)
                       ("mlp.down_proj.weight" ,d ,f)
                       ("input_layernorm.weight" ,d) ("post_attention_layernorm.weight" ,d)))
        (push (cons (format nil "model.layers.~D.~A" i (car entry)) (cdr entry)) schema)))
    schema))
(define-native "tb_weights_load" :pointer (context :pointer) (path :string))
(define-native "tb_weights_next" :int (weights :pointer) (key :pointer) (value :pointer))
(define-native "tb_weights_free" :void (weights :pointer))
(define-native "tb_weights_save" :int (path :string) (names :pointer) (values :pointer) (count :int))
(defun load-weights (path backend destination &key required-dtype preserve-dtype)
  (let ((w (checked-pointer (%tb-weights-load (pointer backend) (namestring path))))
        (*backend* backend) (loaded nil))
    (unwind-protect
         (cffi:with-foreign-objects ((key :pointer) (value :pointer))
           (loop for status = (%tb-weights-next w key value) until (= status 2)
                 do (checked-status status)
                    (let ((name (cffi:foreign-string-to-lisp (cffi:mem-ref key :pointer) :encoding :utf-8)))
                      (with-resource (raw (wrap-tensor (cffi:mem-ref value :pointer) backend))
                        (when (gethash name destination)
                          (fail 'compatibility-error "Duplicate tensor ~A" name))
                        (when (and required-dtype (not (eq (tensor-dtype raw) required-dtype)))
                          (fail 'compatibility-error "Tensor ~A has dtype ~A; expected ~A"
                                name (tensor-dtype raw) required-dtype))
                        (setf (gethash name destination)
                              (if preserve-dtype (retain-tensor raw) (cast-float raw)))
                        (push name loaded)))))
      (%tb-weights-free w))
    (nreverse loaded)))
(defun checkpoint-files (directory)
  (let ((index (merge-pathnames "model.safetensors.index.json" directory))
        (single (merge-pathnames "model.safetensors" directory)))
    (cond ((probe-file index)
           (let ((map (gethash "weight_map" (read-json index))))
             (unless (and (hash-table-p map) (plusp (hash-table-count map)))
               (fail 'compatibility-error "Missing checkpoint shard weight_map"))
             (values
              (loop for file in (remove-duplicates (loop for v being the hash-values of map collect v) :test #'equal)
                    collect (progn
                      (unless (and (stringp file) (equal file (file-namestring file))
                                   (equal (pathname-type file) "safetensors"))
                        (fail 'compatibility-error "Invalid shard filename: ~S" file))
                      (merge-pathnames file directory))) map)))
          ((probe-file single) (values (list single) nil))
          (t (fail 'compatibility-error "No standard safetensors checkpoint in ~A" directory)))))
(defvar *hub-command-runner* #'uiop:run-program)
(defun run-hub-helper (request &key python-executable)
  (let* ((script (merge-pathnames "scripts/hf-artifacts.py"
                                  (asdf:system-source-directory "cl-transformer-blocks")))
         (payload (with-output-to-string (stream)
                    (yason:encode request stream)))
         (output
           (handler-case
               (with-input-from-string (input payload)
                 (funcall *hub-command-runner*
                          (list (or python-executable (default-python-executable))
                                "-u" (namestring script))
                          :input input :output :string :error-output *error-output*))
             (error (condition)
               (fail 'compatibility-error "Hugging Face artifact helper failed: ~A" condition)))))
    (handler-case
        (yason:parse output :json-arrays-as-vectors t :json-booleans-as-symbols t)
      (error (condition)
        (fail 'compatibility-error "Invalid Hugging Face helper response: ~A" condition)))))
(defun download-from-hub (repo-id &key revision (local-files-only nil) cache-directory
                                      python-executable)
  "Resolve a Hub model snapshot through the locked helper and return its local directory."
  (unless (valid-hub-repo-id-p repo-id)
    (fail 'shape-error "Invalid Hugging Face model repository ID"))
  (unless (or (null revision) (stringp revision))
    (fail 'shape-error "REVISION must be NIL or a string"))
  (let ((request (make-hash-table :test 'equal)))
    (setf (gethash "action" request) "download"
          (gethash "repo_id" request) repo-id
          (gethash "revision" request) revision
          (gethash "local_files_only" request) (if local-files-only 'yason:true 'yason:false)
          (gethash "cache_directory" request)
          (and cache-directory (namestring (uiop:ensure-directory-pathname cache-directory))))
    (let* ((response (run-hub-helper request :python-executable python-executable))
           (path (gethash "path" response)))
      (unless (and (stringp path) (uiop:directory-exists-p path))
        (fail 'compatibility-error "Hub helper did not return an existing snapshot directory"))
      (values (truename path) (gethash "resolved_revision" response)))))
(defun from-pretrained (directory &key (device :cpu) model-id revision
                          (execution :native) (task :causal-lm) (local-files-only nil)
                          (trust-remote-code nil) (dtype "auto") python-executable
                          (max-output-elements 2000000) auto-class python-threads
                          (training-dtype "float32") cache-directory)
  "Load a local or Hub native model, or an explicit resident Python Transformers model.
EXECUTION is :NATIVE or :PYTHON. Pin REVISION for reproducible Hub loading."
  (when (eq execution :python)
    (return-from from-pretrained
      (load-python-model (or model-id directory) :device device :revision revision
                         :task task :local-files-only local-files-only
                         :trust-remote-code trust-remote-code :dtype dtype
                         :python-executable python-executable
                         :max-output-elements max-output-elements :auto-class auto-class
                         :python-threads python-threads :training-dtype training-dtype)))
  (unless (eq execution :native)
    (fail 'compatibility-error "EXECUTION must be :NATIVE or :PYTHON"))
  (let* ((local (and directory (probe-file directory)))
         (repo-id (and (not local) (or model-id (and (stringp directory) directory))))
         (resolved-revision revision)
         (directory
           (if local
               (uiop:ensure-directory-pathname local)
               (multiple-value-bind (path resolved)
                   (download-from-hub repo-id :revision revision
                                              :local-files-only local-files-only
                                              :cache-directory cache-directory
                                              :python-executable python-executable)
                 (setf resolved-revision resolved) path)))
         (c (validate-config (read-json (merge-pathnames "config.json" directory))))
         (weights (make-hash-table :test 'equal)) (backend (make-backend :device device))
         (success nil))
    (unwind-protect
         (progn
           (multiple-value-bind (files index) (checkpoint-files directory)
             (dolist (file files)
               (let ((names (load-weights file backend weights)))
                 (when index
                   (dolist (name names)
                     (unless (equal (gethash name index) (file-namestring file))
                       (fail 'compatibility-error "Checkpoint shard index disagrees for tensor ~A" name))))))
             (when (and index (/= (hash-table-count index) (hash-table-count weights)))
               (fail 'compatibility-error "Checkpoint shard index coverage mismatch")))
           (sanitize-architecture-weights (architecture-class c) c weights)
           (let ((schema (parameter-schema c)))
             (dolist (entry schema)
               (let ((value (gethash (car entry) weights)))
                 (unless value (fail 'compatibility-error "Missing tensor: ~A" (car entry)))
                 (unless (equal (tensor-shape value) (cdr entry))
                   (fail 'shape-error "~A: expected ~S, found ~S" (car entry) (cdr entry) (tensor-shape value)))))
             ;; Tied heads are serialized once. A duplicated head must agree exactly.
             (when (and (tied-embeddings-p c) (gethash "lm_head.weight" weights))
               (let ((head (gethash "lm_head.weight" weights)))
                 (unless (equalp (tensor-array head) (tensor-array (gethash (embedding-weight-name (architecture-class c)) weights)))
                   (fail 'compatibility-error "Tied embedding tensors disagree"))
                 (dispose head) (remhash "lm_head.weight" weights)))
             (maphash (lambda (key value) (declare (ignore value))
                        (unless (assoc key schema :test #'equal)
                          (fail 'compatibility-error "Unexpected tensor: ~A" key))) weights))
           (let ((model (make-instance (architecture-class c) :config c :backend backend
                                      :directory directory :parameters weights
                                      :source-id (or repo-id model-id (namestring (truename directory)))
                                      :source-revision resolved-revision)))
             (setf success t) model))
      (unless success
        (maphash (lambda (key value) (declare (ignore key)) (dispose value)) weights)
        (dispose backend)))))
(defun save-weights (parameters path)
  (let ((count (length parameters)) (strings nil))
    (unwind-protect
         (cffi:with-foreign-objects ((names :pointer count) (values :pointer count))
           (loop for (name . value) in parameters for i from 0 do
             (let ((p (cffi:foreign-string-alloc name :encoding :utf-8)))
               (push p strings) (setf (cffi:mem-aref names :pointer i) p
                                      (cffi:mem-aref values :pointer i) (pointer value))))
           (checked-status (%tb-weights-save (namestring path) names values count)))
      (mapc #'cffi:foreign-string-free strings))))
(defun tensor-storage-bytes (tensor)
  (* (reduce #'* (tensor-shape tensor))
     (ecase (tensor-dtype tensor)
       ((:bool :uint8 :int8) 1)
       ((:uint16 :int16 :float16 :bfloat16) 2)
       ((:uint32 :int32 :float32) 4)
       ((:uint64 :int64 :float64 :complex64) 8))))
(defun partition-weight-shards (parameters maximum-bytes)
  (if (null maximum-bytes)
      (list parameters)
      (let ((shards nil) (current nil) (current-bytes 0))
        (dolist (entry parameters)
          (let ((bytes (tensor-storage-bytes (cdr entry))))
            (when (and current (> (+ current-bytes bytes) maximum-bytes))
              (push (nreverse current) shards)
              (setf current nil current-bytes 0))
            (push entry current)
            (incf current-bytes bytes)))
        (when current (push (nreverse current) shards))
        (nreverse shards))))
(defvar *before-directory-publication* (lambda (staging destination)
                                         (declare (ignore staging destination))))
(defun create-publication-stage (destination)
  (let* ((parent (uiop:pathname-parent-directory-pathname destination))
         (leaf (or (car (last (pathname-directory destination))) "checkpoint")))
    (ensure-directories-exist (merge-pathnames ".publication-parent" parent))
    (loop for nonce from 0
          for stage = (merge-pathnames
                       (format nil ".tb-stage-~A-~D-~D/" leaf
                               (get-universal-time) (+ nonce (random 1000000000)))
                       parent)
          for status = (create-directory-exclusively stage)
          when (zerop status) return stage
          unless (= status 1) do (checked-filesystem-status status))))
(defun call-with-staged-directory-publication (destination writer)
  "Build a directory beside DESTINATION and expose it with one filesystem operation."
  (let* ((destination (uiop:ensure-directory-pathname destination))
         (staging (create-publication-stage destination)))
    (unwind-protect
         (progn
           (funcall writer staging)
           (funcall *before-directory-publication* staging destination)
           (publish-directory staging destination)
           destination)
      ;; After an exchange, STAGING names the prior published checkpoint.
      (when (uiop:directory-exists-p staging)
        (uiop:delete-directory-tree staging :validate t :if-does-not-exist :ignore)))))
(defun save-model-weights (parameters destination maximum-bytes)
  (let* ((shards (partition-weight-shards parameters maximum-bytes))
         (count (length shards)))
    (if (= count 1)
        (save-weights (first shards) (merge-pathnames "model.safetensors" destination))
        (let ((index (make-hash-table :test 'equal))
              (metadata (make-hash-table :test 'equal))
              (weight-map (make-hash-table :test 'equal)))
          (loop for shard in shards for number from 1
                for filename = (format nil "model-~5,'0D-of-~5,'0D.safetensors"
                                       number count)
                do (save-weights shard (merge-pathnames filename destination))
                   (dolist (entry shard)
                     (setf (gethash (car entry) weight-map) filename)))
          (setf (gethash "total_size" metadata)
                (loop for entry in parameters sum (tensor-storage-bytes (cdr entry)))
                (gethash "metadata" index) metadata
                (gethash "weight_map" index) weight-map)
          (write-json index (merge-pathnames "model.safetensors.index.json" destination))))))
(defgeneric model-export-config (model))
(defmethod model-export-config ((model model))
  (let ((configuration (alexandria:copy-hash-table (model-config model))))
    (setf (gethash "dtype" configuration) "float32")
    (when (gethash "torch_dtype" configuration)
      (setf (gethash "torch_dtype" configuration) "float32"))
    configuration))
(defun native-model-export-destination (model destination)
  (let ((destination (uiop:ensure-directory-pathname destination)))
    (when (and (model-directory model)
               (uiop:directory-exists-p destination)
               (equal (truename destination) (truename (model-directory model))))
      (fail 'compatibility-error "Export to a separate directory to preserve the source checkpoint"))
    destination))
(defun write-python-architecture-files (model destination)
  (dolist (entry (python-architecture-files model))
    (let ((name (car entry)) (source (cdr entry)))
      (unless (and (stringp name) (equal name (file-namestring name))
                   (equal (pathname-type name) "py") (probe-file source))
        (fail 'compatibility-error "Invalid emitted Python architecture file: ~S" name))
      (uiop:copy-file source (merge-pathnames name destination)))))
(defun write-native-model-directory (model destination max-shard-size)
  (save-model-weights (named-parameters model) destination max-shard-size)
  (write-json (model-export-config model) (merge-pathnames "config.json" destination))
  (write-python-architecture-files model destination)
  (dolist (name '("tokenizer.json" "tokenizer_config.json" "special_tokens_map.json"
                   "added_tokens.json" "tokenizer.model" "vocab.json" "merges.txt"
                   "vocab.txt" "generation_config.json" "chat_template.jinja" "recomposition.json"))
    (let ((source (and (or (null (model-tokenizer-assets model))
                          (member name '("generation_config.json" "recomposition.json") :test #'equal))
                       (model-directory model)
                       (merge-pathnames name (model-directory model)))))
      (when (and source (probe-file source))
        (uiop:copy-file source (merge-pathnames name destination)))))
  (dolist (entry (model-tokenizer-assets model))
    (with-open-file (stream (merge-pathnames (car entry) destination)
                            :direction :output :if-exists :supersede
                            :element-type '(unsigned-byte 8))
      (write-sequence (cdr entry) stream)))
  destination)
(defmethod save-pretrained ((model model) destination &key max-shard-size)
  "Atomically publish standard FP32 safetensors and preserved configuration assets."
  (pointer (model-backend model))
  (unless (or (null max-shard-size) (typep max-shard-size '(integer 1)))
    (fail 'shape-error "MAX-SHARD-SIZE must be NIL or a positive byte count"))
  (when (model-adapter model)
    (fail 'compatibility-error "Use SAVE-ADAPTER or MERGE-ADAPTER before exporting a full model"))
  (let ((destination (native-model-export-destination model destination)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (write-native-model-directory model staging max-shard-size)))))
(defun native-publication-stage (model destination model-card repo-id)
  (let ((destination (save-pretrained model destination)))
    (with-open-file (stream (merge-pathnames "README.md" destination)
                            :direction :output :if-exists :supersede
                            :external-format :utf-8)
      (if model-card
          (write-string model-card stream)
          (format stream "---~%library_name: transformers~%---~%~%# ~A~%~%Exported from Common Lisp with cl-transformer-blocks.~%"
                  (car (last (uiop:split-string repo-id :separator '(#\/)))))))
    destination))
(defun native-publication-files (directory)
  (coerce (sort (mapcar #'file-namestring (uiop:directory-files directory)) #'string<) 'vector))
(defmethod push-to-hub ((model model) repo-id
                        &key (private nil) revision commit-message commit-description
                             (create-pr nil) model-card dry-run-directory)
  "Publish a native standard model through inherited Hugging Face credentials."
  (unless (valid-hub-repo-id-p repo-id)
    (fail 'shape-error "Invalid Hugging Face model repository ID"))
  (when (and model-card (not (stringp model-card)))
    (fail 'shape-error "MODEL-CARD must be NIL or a string"))
  (if dry-run-directory
      (let* ((directory (native-publication-stage model dry-run-directory model-card repo-id))
             (result (make-hash-table :test 'equal)))
        (setf (gethash "repo_id" result) repo-id
              (gethash "dry_run" result) 'yason:true
              (gethash "files" result) (native-publication-files directory))
        result)
      (let* ((temporary (merge-pathnames
                         (format nil "cl-transformer-blocks-hub-~D-~D/"
                                 (get-universal-time) (random 1000000000))
                         (uiop:temporary-directory))))
        (unwind-protect
             (let ((directory (native-publication-stage model temporary model-card repo-id))
                   (request (make-hash-table :test 'equal)))
               (setf (gethash "action" request) "publish"
                     (gethash "repo_id" request) repo-id
                     (gethash "directory" request) (namestring directory)
                     (gethash "private" request) (if private 'yason:true 'yason:false)
                     (gethash "revision" request) revision
                     (gethash "create_pr" request) (if create-pr 'yason:true 'yason:false)
                     (gethash "commit_message" request) commit-message
                     (gethash "commit_description" request) commit-description)
               (run-hub-helper request))
          (when (probe-file temporary)
            (uiop:delete-directory-tree temporary :validate t :if-does-not-exist :ignore))))))
