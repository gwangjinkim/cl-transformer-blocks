(in-package #:tb)

(defgeneric save-composed-pretrained (model destination &key max-shard-size)
  (:documentation "Export an equivalent versioned component architecture with the current base weights.
The qualified conversion is native bias-free Llama with hidden-size = heads * head-dim.
The source is unchanged. Reload the destination to inspect/train its Lisp components.
Active adapters require an explicit merge or removal before conversion."))

(defmethod save-composed-pretrained ((model t) destination &key max-shard-size)
  (declare (ignore model destination max-shard-size))
  (fail 'compatibility-error "No pretrained-to-component conversion for this model class"))

(defun llama-composition-config (model)
  ;; Llama subclasses may change their graph while retaining familiar weight names.
  ;; Dispatch inheritance alone is not evidence of equivalent numerical semantics.
  (unless (eq (class-of model) (find-class 'llama-model))
    (fail 'compatibility-error "Only the exact native Llama class has a qualified component conversion"))
  (let* ((source (validate-config (model-config model)))
         (heads (gethash "num_attention_heads" source))
         (hidden (gethash "hidden_size" source))
         (names (gethash "architectures" source)))
    (unless (and (equal (gethash "model_type" source) "llama")
                 (= hidden (* heads (head-dimension source)))
                 (or (null names)
                     (and (vectorp names) (= (length names) 1)
                          (equal (aref names 0) "LlamaForCausalLM"))))
      (fail 'compatibility-error "Llama head layout or architecture identity cannot be represented by composed v1"))
    (let* ((block (make-transformer-block
                   :attention (make-rotary-attention
                               :heads heads :kv-heads (gethash "num_key_value_heads" source heads)
                               :rope-theta (rope-base source))
                   :feed-forward (make-swiglu :intermediate-size (gethash "intermediate_size" source))
                   :norm-epsilon (gethash "rms_norm_eps" source) :residual :sequential))
           (config (component-object
                    "model_type" "tb_composed" "architectures" #("TBComposedForCausalLM")
                    "composition_version" 1
                    "block_configs" (coerce (loop repeat (gethash "num_hidden_layers" source)
                                                  collect (component-config block)) 'vector)
                    "num_hidden_layers" (gethash "num_hidden_layers" source)
                    "hidden_size" hidden "vocab_size" (gethash "vocab_size" source)
                    "max_position_embeddings" (gethash "max_position_embeddings" source)
                    "rms_norm_eps" (gethash "rms_norm_eps" source)
                    "initializer_range" (gethash "initializer_range" source 0.02)
                    "tie_word_embeddings" (if (tied-embeddings-p source) 'yason:true 'yason:false)
                    "use_cache" 'yason:false "dtype" "float32")))
      (dolist (key '("bos_token_id" "eos_token_id" "pad_token_id"))
        (setf (gethash key config) (copy-component-data (gethash key source))))
      (validate-config config))))

(defun validate-composition-parameters (config parameters)
  (let ((schema (parameter-schema config)))
    (unless (= (length schema) (hash-table-count parameters))
      (fail 'compatibility-error "Source weights do not completely match the component parameter schema"))
    (dolist (entry schema)
      (let ((tensor (gethash (car entry) parameters)))
        (unless tensor (fail 'compatibility-error "Missing component tensor: ~A" (car entry)))
        (unless (equal (tensor-shape tensor) (cdr entry))
          (fail 'shape-error "Component tensor ~A requires shape ~S" (car entry) (cdr entry)))))))

(defun publish-composed-parameters (model destination config parameters max-shard-size &optional provenance)
  (unless (or (null max-shard-size) (typep max-shard-size '(integer 1)))
    (fail 'shape-error "MAX-SHARD-SIZE must be NIL or a positive byte count"))
  (let ((destination (native-model-export-destination model destination)))
    ;; This private, synchronous export view borrows resources from MODEL. It is
    ;; never returned or disposed: no extra weight buffers, host copies or ownership
    ;; transfer are needed. The caller retains MODEL for the duration of the call.
    (let ((view (make-instance 'composed-causal-lm :config config
                              :backend (model-backend model) :parameters parameters
                              :directory (model-directory model)
                              :source-id (model-source-id model)
                              :source-revision (model-source-revision model))))
      (setf (model-tokenizer-assets view) (model-tokenizer-assets model))
      (call-with-staged-directory-publication
       destination
       (lambda (staging)
         (write-native-model-directory view staging max-shard-size)
         (when provenance (write-json provenance (merge-pathnames "recomposition.json" staging)))
         ;; GenerationConfig overrides model.config in Transformers. Preserve its
         ;; sampling/token settings, but select the emitted model's uncached path.
         (let ((path (merge-pathnames "generation_config.json" staging)))
           (when (probe-file path)
             (let ((generation (read-json path)))
               (unless (hash-table-p generation)
                 (fail 'compatibility-error "Generation configuration must be a JSON object"))
               (setf (gethash "use_cache" generation) 'yason:false)
               (write-json generation path)))))))))

(defmethod save-composed-pretrained ((model llama-model) destination &key max-shard-size)
  (pointer (model-backend model))
  (when (model-adapter model)
    (fail 'compatibility-error "Merge or remove the active adapter before converting its base architecture"))
  (let ((config (llama-composition-config model)))
    (validate-composition-parameters config (parameters model))
    (publish-composed-parameters model destination config (parameters model) max-shard-size)))

(defgeneric save-recomposed-pretrained (model destination layer-indices &key max-shard-size)
  (:documentation "Export a new composed stack using zero-based source layer indices in the requested order.
Indices may repeat; each occurrence is serialized as independent trainable parameters.
Embeddings, final normalization, vocabulary head, and tokenizer stay with the source.
This changes the model function. No claim of preserved model quality is made."))
(defmethod save-recomposed-pretrained ((model t) destination layer-indices &key max-shard-size)
  (declare (ignore model destination layer-indices max-shard-size))
  (fail 'compatibility-error "Layer reuse requires a native composed model; convert supported Llama sources first"))

(defun validated-layer-selection (indices count)
  (unless (and (or (and (listp indices) (alexandria:proper-list-p indices)) (vectorp indices))
               (not (stringp indices)) (plusp (length indices))
               (every (lambda (index) (typep index `(integer 0 (,count)))) indices))
    (fail 'compatibility-error "Layer indices must be a nonempty list/vector of integers in [0, ~D)" count))
  (copy-seq (coerce indices 'vector)))

(defun selected-parameter-source (name indices)
  (let ((prefix "model.layers."))
    (if (alexandria:starts-with-subseq prefix name)
        (let* ((end (position #\. name :start (length prefix)))
               (index (parse-integer name :start (length prefix) :end end)))
          (format nil "~A~D~A" prefix (aref indices index) (subseq name end)))
        name)))

(defmethod save-recomposed-pretrained ((model composed-causal-lm) destination layer-indices &key max-shard-size)
  (pointer (model-backend model))
  (unless (eq (class-of model) (find-class 'composed-causal-lm))
    (fail 'compatibility-error "Unknown composed subclasses require their own qualified layer-reuse method"))
  (validate-composition-state model)
  (validate-config (model-config model))
  (when (model-adapter model)
    (fail 'compatibility-error "Merge or remove the active adapter before selecting base layers"))
  (validate-composition-parameters (model-config model) (parameters model))
  (let* ((blocks (gethash "block_configs" (model-config model)))
         (indices (validated-layer-selection layer-indices (length blocks)))
         (config (copy-component-data (model-config model)))
         (weights (make-hash-table :test 'equal)))
    (setf (gethash "block_configs" config)
          (map 'vector (lambda (index) (copy-component-data (aref blocks index))) indices)
          (gethash "num_hidden_layers" config) (length indices))
    (validate-config config)
    (dolist (entry (parameter-schema config))
      (setf (gethash (car entry) weights)
            (gethash (selected-parameter-source (car entry) indices) (parameters model))))
    (validate-composition-parameters config weights)
    (publish-composed-parameters
     model destination config weights max-shard-size
     (component-object "format_version" 1 "source_id" (model-source-id model)
                       "source_revision" (model-source-revision model)
                       "source_layer_count" (length blocks)
                       "source_modified" (if (base-updated-p model) 'yason:true 'yason:false)
                       "layer_indices" indices))))
