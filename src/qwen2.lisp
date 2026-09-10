(in-package #:tb)

(defclass qwen2-model (llama-model) ())
(register-architecture "qwen2" 'qwen2-model)

(defmethod embedding-weight-name ((architecture (eql 'qwen2-model)))
  "model.embed_tokens.weight")
(defmethod default-tie-embeddings ((architecture (eql 'qwen2-model)))
  'yason:false)

(defparameter +qwen2-config-keys+
  '("architectures" "model_type" "vocab_size" "hidden_size" "intermediate_size"
    "num_hidden_layers" "num_attention_heads" "num_key_value_heads" "head_dim"
    "hidden_act" "max_position_embeddings" "initializer_range" "rms_norm_eps"
    "use_cache" "tie_word_embeddings" "rope_theta" "rope_scaling" "rope_parameters"
    "rope_interleaved" "use_mrope" "use_sliding_window" "sliding_window" "max_window_layers"
    "layer_types" "attention_dropout" "bos_token_id" "eos_token_id" "pad_token_id"
    "dtype" "torch_dtype" "transformers_version" "_name_or_path" "_commit_hash"
    "_attn_implementation_autoset" "return_dict" "output_attentions"
    "output_hidden_states" "task_specific_params" "_num_labels" "id2label" "label2id"
    "problem_type" "quantization_config" "auto_map"))

(defmethod validate-architecture-config ((architecture (eql 'qwen2-model)) c)
  (unless (and (hash-table-p c) (equal (gethash "model_type" c) "qwen2"))
    (fail 'compatibility-error "Expected a Qwen2 configuration"))
  (maphash
   (lambda (key value)
     (declare (ignore value))
     (unless (member key +qwen2-config-keys+ :test #'equal)
       (fail 'compatibility-error "Unrecognized Qwen2 configuration field: ~A" key)))
   c)
  (dolist (key '("vocab_size" "hidden_size" "intermediate_size" "num_hidden_layers"
                 "num_attention_heads" "max_position_embeddings"))
    (unless (typep (gethash key c) '(integer 1 2147483647))
      (fail 'compatibility-error "Invalid positive Qwen2 dimension: ~A" key)))
  (let* ((heads (gethash "num_attention_heads" c))
         (kv-heads (gethash "num_key_value_heads" c heads))
         (dimension (head-dimension c)))
    (unless (and (typep kv-heads '(integer 1))
                 (zerop (mod heads kv-heads))
                 (typep dimension '(integer 2))
                 (evenp dimension))
      (fail 'compatibility-error "Invalid Qwen2 attention/grouped-head dimensions")))
  (unless (equal (gethash "hidden_act" c "silu") "silu")
    (fail 'compatibility-error "Only Qwen2 SiLU activation is supported"))
  (unless (zerop (gethash "attention_dropout" c 0))
    (fail 'compatibility-error "Qwen2 attention dropout is unsupported"))
  (when (json-true-p (gethash "use_sliding_window" c))
    (fail 'compatibility-error "Qwen2 sliding-window attention is unsupported"))
  (let ((window (gethash "sliding_window" c)))
    ;; Older official configs preserve this inactive value even when the feature is false.
    (unless (or (null window) (typep window '(integer 1)))
      (fail 'compatibility-error "Invalid Qwen2 sliding-window size")))
  (let ((layers (gethash "layer_types" c)))
    (when layers
      (unless (and (vectorp layers)
                   (= (length layers) (gethash "num_hidden_layers" c))
                   (every (lambda (kind) (equal kind "full_attention")) layers))
        (fail 'compatibility-error "Only full-attention Qwen2 layers are supported"))))
  (dolist (key '("rope_interleaved" "use_mrope" "quantization_config" "auto_map"))
    (let ((value (gethash key c)))
      (when (and value (not (eq value 'yason:false)))
        (fail 'compatibility-error "Unsupported Qwen2 feature: ~A" key))))
  (let ((scaling (gethash "rope_scaling" c))
        (parameters (gethash "rope_parameters" c)))
    (when parameters
      (unless (hash-table-p parameters)
        (fail 'compatibility-error "Invalid Qwen2 rope_parameters"))
      (maphash
       (lambda (key value)
         (declare (ignore value))
         (unless (member key '("rope_type" "rope_theta") :test #'equal)
           (fail 'compatibility-error "Unsupported Qwen2 RoPE parameter: ~A" key)))
       parameters))
    (when (or scaling
              (and parameters
                   (not (equal (gethash "rope_type" parameters "default") "default"))))
      (fail 'compatibility-error "Only unscaled default Qwen2 RoPE is supported")))
  (unless (and (realp (rope-base c)) (> (rope-base c) 0)
               (realp (gethash "rms_norm_eps" c))
               (> (gethash "rms_norm_eps" c) 0))
    (fail 'compatibility-error "Invalid Qwen2 RoPE base or normalization epsilon"))
  c)

(defmethod architecture-parameter-schema ((architecture (eql 'qwen2-model)) c)
  "Canonical Qwen2 causal-LM weights. Linear matrices are stored as (output,input)."
  (let* ((hidden (gethash "hidden_size" c))
         (intermediate (gethash "intermediate_size" c))
         (vocabulary (gethash "vocab_size" c))
         (heads (gethash "num_attention_heads" c))
         (kv-heads (gethash "num_key_value_heads" c heads))
         (head-dimension (head-dimension c))
         (schema (list (list "model.embed_tokens.weight" vocabulary hidden)
                       (list "model.norm.weight" hidden))))
    (unless (tied-embeddings-p c)
      (push (list "lm_head.weight" vocabulary hidden) schema))
    (dotimes (layer (gethash "num_hidden_layers" c))
      (dolist (entry `(("self_attn.q_proj.weight" ,(* heads head-dimension) ,hidden)
                       ("self_attn.q_proj.bias" ,(* heads head-dimension))
                       ("self_attn.k_proj.weight" ,(* kv-heads head-dimension) ,hidden)
                       ("self_attn.k_proj.bias" ,(* kv-heads head-dimension))
                       ("self_attn.v_proj.weight" ,(* kv-heads head-dimension) ,hidden)
                       ("self_attn.v_proj.bias" ,(* kv-heads head-dimension))
                       ("self_attn.o_proj.weight" ,hidden ,(* heads head-dimension))
                       ("mlp.gate_proj.weight" ,intermediate ,hidden)
                       ("mlp.up_proj.weight" ,intermediate ,hidden)
                       ("mlp.down_proj.weight" ,hidden ,intermediate)
                       ("input_layernorm.weight" ,hidden)
                       ("post_attention_layernorm.weight" ,hidden)))
        (push (cons (format nil "model.layers.~D.~A" layer (car entry)) (cdr entry))
              schema)))
    schema))

(defun optional-weight (model name)
  (multiple-value-bind (value present) (and *parameter-overrides*
                                             (gethash name *parameter-overrides*))
    (if present
        value
        (gethash name (parameters model)))))

(defmethod project ((model qwen2-model) inputs parameter-name)
  "Apply Qwen2's learned Q/K/V bias while retaining the common projection graph."
  (let* ((projected (call-next-method))
         (bias-name (concatenate 'string
                                 (subseq parameter-name 0 (- (length parameter-name) 6))
                                 "bias"))
         (bias (optional-weight model bias-name)))
    (if bias (add projected bias) projected)))
