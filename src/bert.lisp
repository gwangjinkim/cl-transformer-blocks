(in-package #:tb)

(defclass bert-masked-lm (model) ())
(register-architecture "bert" 'bert-masked-lm)

(defmethod architecture-capabilities ((architecture (eql 'bert-masked-lm)) config directory)
  (declare (ignore architecture directory))
  (let ((training (and (zerop (gethash "hidden_dropout_prob" config 0.0))
                       (zerop (gethash "attention_probs_dropout_prob" config 0.0)))))
    (list :operations (append '(:forward :export :lora :rslora :token-type-ids)
                              (and training '(:training :training-checkpoint)))
          :mask-policy :right-padding
          :restrictions
          '("FP32 execution and export"
            "masked-language-model head"
            "absolute positions; token-type IDs default to zero"
            "training requires zero dropout"
            "KV caching and generation unavailable"))))

(defmethod embedding-weight-name ((architecture (eql 'bert-masked-lm)))
  "bert.embeddings.word_embeddings.weight")
(defmethod default-tie-embeddings ((architecture (eql 'bert-masked-lm)))
  'yason:true)
(defmethod model-layer-count ((model bert-masked-lm))
  (config model "num_hidden_layers"))
(defmethod model-context-length ((model bert-masked-lm))
  (config model "max_position_embeddings"))
(defmethod make-cache ((model bert-masked-lm) &key growth-step)
  (declare (ignore model growth-step))
  (fail 'compatibility-error "Native bidirectional BERT does not use a KV cache"))
(defmethod validate-training-config ((model bert-masked-lm))
  (dolist (key '("hidden_dropout_prob" "attention_probs_dropout_prob"))
    (unless (zerop (config model key 0.1))
      (fail 'compatibility-error
            "Native BERT training requires zero configured dropout (~A)" key))))

(defmethod validate-token-type-ids ((model bert-masked-lm) inputs token-type-ids)
  (when token-type-ids
    (unless (and (arrayp token-type-ids)
                 (equal (array-dimensions token-type-ids) (array-dimensions inputs)))
      (fail 'shape-error "BERT token-type IDs must have the input shape"))
    (dotimes (i (array-total-size token-type-ids))
      (unless (typep (row-major-aref token-type-ids i)
                     `(integer 0 (,(config model "type_vocab_size"))))
        (fail 'shape-error "BERT token-type ID outside type_vocab_size")))))

(defmethod prepare-training-batch ((model bert-masked-lm) ids labels mask)
  "Select explicitly labeled token positions; BERT labels are not shifted."
  (pointer (model-backend model))
  (validate-training-config model)
  (validate-inputs model ids mask nil)
  (let ((batch (array-dimension ids 0))
        (steps (array-dimension ids 1))
        (targets nil)
        (selected nil))
    (unless (and labels (arrayp labels)
                 (equal (array-dimensions labels) (list batch steps)))
      (fail 'shape-error "BERT labels must have the input shape; -100 ignores a token"))
    (dotimes (i (array-total-size labels))
      (let ((target (row-major-aref labels i)))
        (unless (or (eql target -100)
                    (typep target `(integer 0 (,(config model "vocab_size")))))
          (fail 'shape-error "BERT label out of vocabulary"))
        (unless (eql target -100)
          (push target targets)
          (push i selected))))
    (unless targets
      (fail 'shape-error "BERT training requires at least one supervised token"))
    (values ids mask (coerce (nreverse targets) 'vector)
            (coerce (nreverse selected) 'vector))))

(defparameter +bert-config-keys+
  '("model_type" "architectures" "vocab_size" "hidden_size" "num_hidden_layers"
    "num_attention_heads" "intermediate_size" "hidden_act" "hidden_dropout_prob"
    "attention_probs_dropout_prob" "max_position_embeddings" "type_vocab_size"
    "initializer_range" "layer_norm_eps" "pad_token_id" "use_cache"
    "classifier_dropout" "is_decoder" "add_cross_attention" "bos_token_id"
    "eos_token_id" "tie_word_embeddings" "position_embedding_type"
    "chunk_size_feed_forward" "gradient_checkpointing" "dtype" "torch_dtype" "transformers_version"
    "_name_or_path" "_commit_hash" "_attn_implementation_autoset" "return_dict"
    "output_attentions" "output_hidden_states" "task_specific_params" "_num_labels"
    "id2label" "label2id" "problem_type" "quantization_config" "auto_map"))

(defmethod validate-architecture-config ((architecture (eql 'bert-masked-lm)) c)
  (unless (and (hash-table-p c) (equal (gethash "model_type" c) "bert"))
    (fail 'compatibility-error "Expected a BERT configuration"))
  (maphash
   (lambda (key value)
     (declare (ignore value))
     (unless (member key +bert-config-keys+ :test #'equal)
       (fail 'compatibility-error "Unrecognized BERT configuration field: ~A" key)))
   c)
  (let ((architectures (gethash "architectures" c)))
    (unless (and (vectorp architectures)
                 (find "BertForMaskedLM" architectures :test #'equal))
      (fail 'compatibility-error "Native BERT currently requires BertForMaskedLM")))
  (dolist (key '("vocab_size" "hidden_size" "num_hidden_layers" "num_attention_heads"
                 "intermediate_size" "max_position_embeddings" "type_vocab_size"))
    (unless (typep (gethash key c) '(integer 1 2147483647))
      (fail 'compatibility-error "Invalid positive BERT dimension: ~A" key)))
  (unless (zerop (mod (gethash "hidden_size" c) (gethash "num_attention_heads" c)))
    (fail 'compatibility-error "BERT hidden size must divide evenly across attention heads"))
  (unless (equal (gethash "hidden_act" c "gelu") "gelu")
    (fail 'compatibility-error "Only exact BERT GELU is supported"))
  (unless (equal (gethash "position_embedding_type" c "absolute") "absolute")
    (fail 'compatibility-error "Only absolute BERT position embeddings are supported"))
  (unless (tied-embeddings-p c)
    (fail 'compatibility-error "Native BERT currently requires tied decoder embeddings"))
  (dolist (key '("is_decoder" "add_cross_attention" "gradient_checkpointing"
                 "quantization_config" "auto_map"))
    (let ((value (gethash key c)))
      (when (and value (not (eq value 'yason:false)))
        (fail 'compatibility-error "Unsupported BERT feature: ~A" key))))
  (dolist (key '("hidden_dropout_prob" "attention_probs_dropout_prob"))
    (let ((probability (gethash key c 0.1)))
      (unless (and (realp probability) (<= 0 probability) (< probability 1))
        (fail 'compatibility-error "Invalid BERT dropout: ~A" key))))
  (unless (and (realp (gethash "layer_norm_eps" c))
               (> (gethash "layer_norm_eps" c) 0)
               (zerop (gethash "chunk_size_feed_forward" c 0)))
    (fail 'compatibility-error "Invalid BERT normalization or unsupported feed-forward chunking"))
  c)

(defmethod sanitize-architecture-weights ((architecture (eql 'bert-masked-lm)) config weights)
  "Canonicalize historical LayerNorm names and discard a checked NSP/pooler group."
  (dolist (name (loop for key being the hash-keys of weights collect key))
    (let ((suffix (cond ((alexandria:ends-with-subseq ".LayerNorm.gamma" name)
                         ".LayerNorm.gamma")
                        ((alexandria:ends-with-subseq ".LayerNorm.beta" name)
                         ".LayerNorm.beta"))))
      (when suffix
        (let ((canonical
                (concatenate 'string
                             (subseq name 0 (- (length name) (length suffix)))
                             (if (equal suffix ".LayerNorm.gamma")
                                 ".LayerNorm.weight" ".LayerNorm.bias"))))
          (when (gethash canonical weights)
            (fail 'compatibility-error
                  "Both legacy and canonical BERT LayerNorm tensors are present: ~A" canonical))
          (setf (gethash canonical weights) (gethash name weights))
          (remhash name weights)))))
  (let* ((hidden (gethash "hidden_size" config))
         (extras `(("bert.pooler.dense.weight" ,hidden ,hidden)
                   ("bert.pooler.dense.bias" ,hidden)
                   ("cls.seq_relationship.weight" 2 ,hidden)
                   ("cls.seq_relationship.bias" 2)))
         (present (count-if (lambda (entry) (gethash (car entry) weights)) extras)))
    (when (plusp present)
      (unless (= present (length extras))
        (fail 'compatibility-error "Incomplete BERT pretraining-only tensor group"))
      (dolist (entry extras)
        (let ((value (gethash (car entry) weights)))
          (unless (equal (tensor-shape value) (cdr entry))
            (fail 'shape-error "Invalid BERT pretraining-only tensor: ~A" (car entry)))
          (dispose value)
          (remhash (car entry) weights))))))

(defmethod architecture-parameter-schema ((architecture (eql 'bert-masked-lm)) c)
  (let* ((hidden (gethash "hidden_size" c))
         (intermediate (gethash "intermediate_size" c))
         (vocabulary (gethash "vocab_size" c))
         (schema `(("bert.embeddings.word_embeddings.weight" ,vocabulary ,hidden)
                   ("bert.embeddings.position_embeddings.weight"
                    ,(gethash "max_position_embeddings" c) ,hidden)
                   ("bert.embeddings.token_type_embeddings.weight"
                    ,(gethash "type_vocab_size" c) ,hidden)
                   ("bert.embeddings.LayerNorm.weight" ,hidden)
                   ("bert.embeddings.LayerNorm.bias" ,hidden)
                   ("cls.predictions.bias" ,vocabulary)
                   ("cls.predictions.transform.dense.weight" ,hidden ,hidden)
                   ("cls.predictions.transform.dense.bias" ,hidden)
                   ("cls.predictions.transform.LayerNorm.weight" ,hidden)
                   ("cls.predictions.transform.LayerNorm.bias" ,hidden))))
    (unless (tied-embeddings-p c)
      (push (list "cls.predictions.decoder.weight" vocabulary hidden) schema))
    (dotimes (layer (gethash "num_hidden_layers" c))
      (dolist (entry `(("attention.self.query.weight" ,hidden ,hidden)
                       ("attention.self.query.bias" ,hidden)
                       ("attention.self.key.weight" ,hidden ,hidden)
                       ("attention.self.key.bias" ,hidden)
                       ("attention.self.value.weight" ,hidden ,hidden)
                       ("attention.self.value.bias" ,hidden)
                       ("attention.output.dense.weight" ,hidden ,hidden)
                       ("attention.output.dense.bias" ,hidden)
                       ("attention.output.LayerNorm.weight" ,hidden)
                       ("attention.output.LayerNorm.bias" ,hidden)
                       ("intermediate.dense.weight" ,intermediate ,hidden)
                       ("intermediate.dense.bias" ,intermediate)
                       ("output.dense.weight" ,hidden ,intermediate)
                       ("output.dense.bias" ,hidden)
                       ("output.LayerNorm.weight" ,hidden)
                       ("output.LayerNorm.bias" ,hidden)))
        (push (cons (format nil "bert.encoder.layer.~D.~A" layer (car entry))
                    (cdr entry))
              schema)))
    schema))

(defun bert-linear (model input weight-name)
  (add (project model input weight-name)
       (weight model (concatenate 'string
                                  (subseq weight-name 0 (- (length weight-name) 6))
                                  "bias"))))

(defun bidirectional-mask (batch steps padding)
  (let ((mask (make-array (list batch 1 steps steps) :element-type 'single-float)))
    (dotimes (b batch)
      (dotimes (query steps)
        (dotimes (key steps)
          (setf (aref mask b 0 query key)
                (if (or (null padding) (= 1 (aref padding b key)))
                    0.0
                    most-negative-single-float)))))
    (tensor-from-array *backend* mask)))

(defclass bert-layer () ((index :initarg :index :reader bert-layer-index)))

(defmethod apply-layer ((layer bert-layer) (model bert-masked-lm)
                        hidden mask offset cache)
  (declare (ignore offset cache))
  (let* ((index (bert-layer-index layer))
         (hidden-size (config model "hidden_size"))
         (heads (config model "num_attention_heads"))
         (head-size (/ hidden-size heads))
         (batch (first (tensor-shape hidden)))
         (steps (second (tensor-shape hidden)))
         (epsilon (config model "layer_norm_eps")))
    (flet ((name (suffix) (format nil "bert.encoder.layer.~D.~A" index suffix))
           (split-heads (tensor)
             (permute (reshape tensor (list batch steps heads head-size)) '(0 2 1 3))))
      (let* ((query (split-heads (bert-linear model hidden (name "attention.self.query.weight"))))
             (key (split-heads (bert-linear model hidden (name "attention.self.key.weight"))))
             (value (split-heads (bert-linear model hidden (name "attention.self.value.weight"))))
             (attended (attention query key value (/ 1.0 (sqrt (float head-size 1.0))) mask))
             (joined (reshape (permute attended '(0 2 1 3))
                              (list batch steps hidden-size)))
             (attention-output
               (layer-norm
                (add hidden
                     (bert-linear model joined (name "attention.output.dense.weight")))
                (weight model (name "attention.output.LayerNorm.weight"))
                (weight model (name "attention.output.LayerNorm.bias")) epsilon))
             (intermediate
               (gelu (bert-linear model attention-output
                                  (name "intermediate.dense.weight")) nil))
             (output
               (layer-norm
                (add attention-output
                     (bert-linear model intermediate (name "output.dense.weight")))
                (weight model (name "output.LayerNorm.weight"))
                (weight model (name "output.LayerNorm.bias")) epsilon)))
        (values output nil)))))

(defmethod compute-logits ((model bert-masked-lm) ids padding offset &optional cache token-type-ids)
  (when (or cache (not (zerop offset)))
    (fail 'compatibility-error "Native bidirectional BERT does not support cached decoding"))
  (let* ((shape (tensor-shape ids))
         (batch (first shape))
         (steps (second shape))
         (positions (make-array steps))
         (token-types (or token-type-ids
                          (tensor-from-array *backend*
                            (make-array (list batch steps) :initial-element 0) :dtype :int32))))
    (dotimes (i steps) (setf (aref positions i) i))
    (let* ((hidden
             (add
              (add (take-indices (weight model "bert.embeddings.word_embeddings.weight") ids)
                   (take-indices
                    (weight model "bert.embeddings.token_type_embeddings.weight")
                    token-types))
              (take-indices
               (weight model "bert.embeddings.position_embeddings.weight")
               (tensor-from-array *backend* positions :dtype :int32))))
           (hidden
             (layer-norm hidden
                         (weight model "bert.embeddings.LayerNorm.weight")
                         (weight model "bert.embeddings.LayerNorm.bias")
                         (config model "layer_norm_eps")))
           (mask (bidirectional-mask batch steps padding)))
      (dotimes (i (model-layer-count model))
        (setf hidden (apply-layer (make-instance 'bert-layer :index i)
                                  model hidden mask 0 nil)))
      (let* ((transformed
               (gelu (bert-linear model hidden
                                  "cls.predictions.transform.dense.weight") nil))
             (transformed
               (layer-norm transformed
                           (weight model "cls.predictions.transform.LayerNorm.weight")
                           (weight model "cls.predictions.transform.LayerNorm.bias")
                           (config model "layer_norm_eps")))
             (decoder (weight model (if (tied-embeddings-p (model-config model))
                                        "bert.embeddings.word_embeddings.weight"
                                        "cls.predictions.decoder.weight"))))
        (values (add (linear transformed decoder)
                     (weight model "cls.predictions.bias"))
                (make-array 0))))))
