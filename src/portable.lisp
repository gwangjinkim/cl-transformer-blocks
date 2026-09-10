(in-package #:tb)

(defclass portable-causal-lm (llama-model) ())
(register-architecture "tb_parallel" 'portable-causal-lm)

(defparameter +portable-auto-map+
  '("AutoConfig" . "configuration_tb_parallel.TBParallelConfig"))
(defparameter +portable-model-auto-map+
  '("AutoModelForCausalLM" . "modeling_tb_parallel.TBParallelForCausalLM"))
(defparameter +portable-config-keys+
  '("architectures" "model_type" "portable_architecture_version" "parallel_residual"
    "vocab_size" "hidden_size" "intermediate_size" "num_hidden_layers"
    "num_attention_heads" "num_key_value_heads" "head_dim" "max_position_embeddings"
    "rms_norm_eps" "rope_theta" "hidden_act" "attention_bias" "mlp_bias"
    "attention_dropout" "tie_word_embeddings" "use_cache" "bos_token_id"
    "eos_token_id" "pad_token_id" "initializer_range" "dtype" "torch_dtype"
    "auto_map" "transformers_version" "_name_or_path" "_commit_hash"
    "return_dict" "output_attentions" "output_hidden_states"))

(defun portable-auto-map ()
  (let ((mapping (make-hash-table :test 'equal)))
    (setf (gethash (car +portable-auto-map+) mapping) (cdr +portable-auto-map+)
          (gethash (car +portable-model-auto-map+) mapping) (cdr +portable-model-auto-map+))
    mapping))

(defun valid-portable-auto-map-p (mapping)
  (and (hash-table-p mapping)
       (= (hash-table-count mapping) 2)
       (equal (gethash (car +portable-auto-map+) mapping) (cdr +portable-auto-map+))
       (equal (gethash (car +portable-model-auto-map+) mapping)
              (cdr +portable-model-auto-map+))))

(defun make-portable-config (&key vocab-size hidden-size intermediate-size layers
                                  attention-heads kv-heads context-length
                                  (rms-norm-epsilon 1e-5) (rope-theta 10000.0)
                                  (tie-word-embeddings nil) (initializer-range 0.02)
                                  (bos-token-id 1) (eos-token-id 2) (pad-token-id 0))
  "Build versioned configuration for the Lisp-defined parallel-residual causal LM."
  (let ((configuration (make-hash-table :test 'equal)))
    (setf (gethash "architectures" configuration) (vector "TBParallelForCausalLM")
          (gethash "model_type" configuration) "tb_parallel"
          (gethash "portable_architecture_version" configuration) 1
          (gethash "parallel_residual" configuration) 'yason:true
          (gethash "vocab_size" configuration) vocab-size
          (gethash "hidden_size" configuration) hidden-size
          (gethash "intermediate_size" configuration) intermediate-size
          (gethash "num_hidden_layers" configuration) layers
          (gethash "num_attention_heads" configuration) attention-heads
          (gethash "num_key_value_heads" configuration) kv-heads
          (gethash "head_dim" configuration)
          (and (typep hidden-size '(integer 1))
               (typep attention-heads '(integer 1))
               (zerop (mod hidden-size attention-heads))
               (/ hidden-size attention-heads))
          (gethash "max_position_embeddings" configuration) context-length
          (gethash "rms_norm_eps" configuration) rms-norm-epsilon
          (gethash "rope_theta" configuration) rope-theta
          (gethash "hidden_act" configuration) "silu"
          (gethash "attention_bias" configuration) 'yason:false
          (gethash "mlp_bias" configuration) 'yason:false
          (gethash "attention_dropout" configuration) 0.0
          (gethash "tie_word_embeddings" configuration)
          (if tie-word-embeddings 'yason:true 'yason:false)
          (gethash "use_cache" configuration) 'yason:false
          (gethash "bos_token_id" configuration) bos-token-id
          (gethash "eos_token_id" configuration) eos-token-id
          (gethash "pad_token_id" configuration) pad-token-id
          (gethash "initializer_range" configuration) initializer-range
          (gethash "dtype" configuration) "float32")
    (validate-config configuration)))

(defmethod validate-architecture-config ((architecture (eql 'portable-causal-lm)) config)
  (declare (ignore architecture))
  (unless (and (hash-table-p config) (equal (gethash "model_type" config) "tb_parallel"))
    (fail 'compatibility-error "Only the registered TB parallel architecture is supported"))
  (maphash (lambda (key value)
             (declare (ignore value))
             (unless (member key +portable-config-keys+ :test #'equal)
               (fail 'compatibility-error
                     "Unrecognized TB parallel configuration field: ~A" key)))
           config)
  (unless (and (eql (gethash "portable_architecture_version" config) 1)
               (json-true-p (gethash "parallel_residual" config)))
    (fail 'compatibility-error "Unsupported portable architecture version or residual topology"))
  (let ((architectures (gethash "architectures" config)))
    (unless (and (vectorp architectures)
                 (= (length architectures) 1)
                 (equal (aref architectures 0) "TBParallelForCausalLM"))
      (fail 'compatibility-error "Portable architecture must identify TBParallelForCausalLM")))
  (dolist (key '("vocab_size" "hidden_size" "intermediate_size" "num_hidden_layers"
                 "num_attention_heads" "num_key_value_heads" "head_dim"
                 "max_position_embeddings"))
    (unless (typep (gethash key config) '(integer 1 2147483647))
      (fail 'compatibility-error "Invalid positive portable dimension: ~A" key)))
  (let ((hidden (gethash "hidden_size" config))
        (heads (gethash "num_attention_heads" config))
        (kv-heads (gethash "num_key_value_heads" config))
        (head-dimension (gethash "head_dim" config)))
    (unless (and (zerop (mod heads kv-heads))
                 (= hidden (* heads head-dimension))
                 (evenp head-dimension))
      (fail 'compatibility-error "Invalid portable attention/grouped-head dimensions")))
  (unless (and (equal (gethash "hidden_act" config) "silu")
               (not (json-true-p (gethash "attention_bias" config)))
               (not (json-true-p (gethash "mlp_bias" config)))
               (zerop (gethash "attention_dropout" config 0))
               (not (json-true-p (gethash "use_cache" config))))
    (fail 'compatibility-error "Portable v1 requires SiLU, bias-free projections, zero dropout, and uncached Python execution"))
  (unless (and (realp (gethash "rms_norm_eps" config))
               (> (gethash "rms_norm_eps" config) 0)
               (realp (gethash "rope_theta" config))
               (> (gethash "rope_theta" config) 0))
    (fail 'compatibility-error "Invalid portable normalization epsilon or RoPE base"))
  (let ((mapping (gethash "auto_map" config)))
    (when (and mapping (not (valid-portable-auto-map-p mapping)))
      (fail 'compatibility-error "Portable auto_map does not name the emitted trusted implementation")))
  config)

(defmethod embedding-weight-name ((architecture (eql 'portable-causal-lm)))
  "model.embed_tokens.weight")
(defmethod default-tie-embeddings ((architecture (eql 'portable-causal-lm)))
  'yason:false)

(defmethod architecture-parameter-schema ((architecture (eql 'portable-causal-lm)) config)
  (declare (ignore architecture))
  (let* ((hidden (gethash "hidden_size" config))
         (intermediate (gethash "intermediate_size" config))
         (vocabulary (gethash "vocab_size" config))
         (heads (gethash "num_attention_heads" config))
         (kv-heads (gethash "num_key_value_heads" config))
         (head-dimension (gethash "head_dim" config))
         (schema (list (cons "model.embed_tokens.weight" (list vocabulary hidden))
                       (cons "model.norm.weight" (list hidden)))))
    (unless (tied-embeddings-p config)
      (push (cons "lm_head.weight" (list vocabulary hidden)) schema))
    (dotimes (layer (gethash "num_hidden_layers" config))
      (dolist (entry `(("self_attn.q_proj.weight" ,(* heads head-dimension) ,hidden)
                       ("self_attn.k_proj.weight" ,(* kv-heads head-dimension) ,hidden)
                       ("self_attn.v_proj.weight" ,(* kv-heads head-dimension) ,hidden)
                       ("self_attn.o_proj.weight" ,hidden ,(* heads head-dimension))
                       ("mlp.gate_proj.weight" ,intermediate ,hidden)
                       ("mlp.up_proj.weight" ,intermediate ,hidden)
                       ("mlp.down_proj.weight" ,hidden ,intermediate)
                       ("input_layernorm.weight" ,hidden)))
        (push (cons (format nil "model.layers.~D.~A" layer (car entry)) (cdr entry))
              schema)))
    schema))

(defmethod architecture-capabilities ((architecture (eql 'portable-causal-lm)) config directory)
  (declare (ignore architecture config directory))
  (list :operations '(:forward :training :training-checkpoint :export :cache :generation
                       :custom-python-export)
        :mask-policy :right-padding
        :restrictions
        '("FP32 native execution and export"
          "unscaled non-interleaved RoPE"
          "zero-dropout parallel residual blocks"
          "emitted Python code requires trust_remote_code=True"
          "emitted Python v1 recomputes generation prefixes")))

(defmethod apply-layer ((layer llama-layer) (model portable-causal-lm)
                        hidden mask offset cache)
  "Parallel residual block: both attention and MLP consume the same normalized input."
  (let* ((configuration (model-config model))
         (heads (config model "num_attention_heads"))
         (kv-heads (config model "num_key_value_heads"))
         (head-dimension (config model "head_dim"))
         (epsilon (config model "rms_norm_eps"))
         (batch (first (tensor-shape hidden)))
         (steps (second (tensor-shape hidden))))
    (flet ((project-layer (input suffix)
             (project model input
                      (format nil "model.layers.~D.~A" (layer-index layer) suffix)))
           (layer-weight (suffix)
             (weight model (format nil "model.layers.~D.~A" (layer-index layer) suffix)))
           (split-heads (value count)
             (permute (reshape value (list batch steps count head-dimension)) '(0 2 1 3))))
      (let* ((normalized (rms-norm hidden (layer-weight "input_layernorm.weight") epsilon))
             (query (rope (split-heads
                           (project-layer normalized "self_attn.q_proj.weight") heads)
                          head-dimension (rope-base configuration) offset))
             (key (rope (split-heads
                         (project-layer normalized "self_attn.k_proj.weight") kv-heads)
                        head-dimension (rope-base configuration) offset))
             (value (split-heads
                     (project-layer normalized "self_attn.v_proj.weight") kv-heads))
             (entry (and cache (aref (cache-entries cache) (layer-index layer))))
             (next-entry nil))
        (when cache
          (multiple-value-bind (used storage)
              (update-cache-tensor cache (and entry (first entry)) key offset)
            (setf key used next-entry (list storage)))
          (multiple-value-bind (used storage)
              (update-cache-tensor cache (and entry (second entry)) value offset)
            (setf value used next-entry (nconc next-entry (list storage)))))
        (let* ((attended (attention query key value
                                    (/ 1.0 (sqrt (float head-dimension 1.0))) mask))
               (joined (reshape (permute attended '(0 2 1 3))
                                (list batch steps (* heads head-dimension))))
               (attention-output
                 (project-layer joined "self_attn.o_proj.weight"))
               (gate (project-layer normalized "mlp.gate_proj.weight"))
               (up (project-layer normalized "mlp.up_proj.weight"))
               (mlp-output
                 (project-layer (multiply (multiply gate (sigmoid gate)) up)
                                "mlp.down_proj.weight")))
          (values (add (add hidden attention-output) mlp-output) next-entry))))))

(defmethod model-export-config ((model portable-causal-lm))
  (let ((configuration (call-next-method)))
    (setf (gethash "auto_map" configuration) (portable-auto-map))
    configuration))

(defmethod python-architecture-files ((model portable-causal-lm))
  (declare (ignore model))
  (let ((directory (asdf:system-relative-pathname
                    "cl-transformer-blocks" "python/tb_parallel/")))
    (list (cons "configuration_tb_parallel.py"
                (merge-pathnames "configuration_tb_parallel.py" directory))
          (cons "modeling_tb_parallel.py"
                (merge-pathnames "modeling_tb_parallel.py" directory)))))

(defun portable-random-parameter-arrays (configuration seed)
  (unless (typep seed '(integer 1 2147483646))
    (fail 'shape-error "Portable initialization seed must be between 1 and 2147483646"))
  (let ((state seed)
        (standard-deviation (coerce (gethash "initializer_range" configuration 0.02)
                                    'double-float)))
    (labels ((uniform ()
               (setf state (mod (* state 48271) 2147483647))
               (/ (coerce state 'double-float) 2147483647.0d0))
             (normal ()
               (* standard-deviation
                  (sqrt (* -2.0d0 (log (max least-positive-double-float (uniform)))))
                  (cos (* 2.0d0 pi (uniform)))))
             (parameter-array (name shape)
               (let ((array (make-array shape :element-type 'single-float)))
                 (dotimes (index (array-total-size array))
                   (setf (row-major-aref array index)
                         (if (search "norm.weight" name)
                             1.0f0
                             (coerce (normal) 'single-float))))
                 array)))
      (loop for (name . shape) in (parameter-schema configuration)
            collect (cons name (parameter-array name shape))))))

(defun make-portable-causal-lm (&key vocab-size hidden-size intermediate-size layers
                                     attention-heads kv-heads context-length
                                     (rms-norm-epsilon 1e-5) (rope-theta 10000.0)
                                     (tie-word-embeddings nil) (initializer-range 0.02)
                                     (bos-token-id 1) (eos-token-id 2) (pad-token-id 0)
                                     (seed 1) (device :cpu))
  "Create a deterministically initialized Lisp-defined causal LM on DEVICE."
  (let* ((configuration
           (make-portable-config
            :vocab-size vocab-size :hidden-size hidden-size
            :intermediate-size intermediate-size :layers layers
            :attention-heads attention-heads :kv-heads kv-heads
            :context-length context-length :rms-norm-epsilon rms-norm-epsilon
            :rope-theta rope-theta :tie-word-embeddings tie-word-embeddings
            :initializer-range initializer-range :bos-token-id bos-token-id
            :eos-token-id eos-token-id :pad-token-id pad-token-id))
         (arrays (portable-random-parameter-arrays configuration seed))
         (backend (make-backend :device device))
         (parameters (make-hash-table :test 'equal))
         (success nil))
    (unwind-protect
         (progn
           (dolist (entry arrays)
             (setf (gethash (car entry) parameters)
                   (tensor-from-array backend (cdr entry))))
           (let ((model (make-instance 'portable-causal-lm
                                       :config configuration :backend backend
                                       :directory nil :parameters parameters
                                       :source-id "common-lisp:tb-parallel"
                                       :source-revision nil)))
             (setf success t)
             model))
      (unless success
        (maphash (lambda (name value)
                   (declare (ignore name))
                   (dispose value))
                 parameters)
        (dispose backend)))))
