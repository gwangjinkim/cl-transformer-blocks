(in-package #:tb)

(defclass python-model ()
  ((process :initarg :process :accessor worker-process)
   (input :initarg :input :reader worker-input)
   (output :initarg :output :reader worker-output)
   (config :initarg :config :reader model-config)
   (info :initarg :info :reader python-worker-info)
   (source :initarg :source :reader python-model-source)
   (revision :initarg :revision :reader model-source-revision)
   (request-id :initform 1 :accessor worker-request-id)
   (disposed :initform nil :accessor python-model-disposed-p)))

(defclass python-tensor ()
  ((shape :initarg :shape :reader tensor-shape)
   (dtype :initarg :dtype :reader tensor-dtype)
   (array :initarg :array :accessor python-tensor-array)))

(defclass python-file-input ()
  ((path :initarg :path :reader python-file-path)
   (media-type :initarg :media-type :reader python-file-media-type)
   (sampling-rate :initarg :sampling-rate :reader python-file-sampling-rate)
   (num-frames :initarg :num-frames :reader python-file-num-frames)
   (fps :initarg :fps :reader python-file-fps)))

(defun make-python-file-input (path media-type &key sampling-rate num-frames fps)
  "Describe a local image, audio, or video file for a Python AutoProcessor."
  (unless (member media-type '(:image :audio :video))
    (fail 'shape-error "MEDIA-TYPE must be :IMAGE, :AUDIO, or :VIDEO"))
  (unless (or (null sampling-rate) (typep sampling-rate '(integer 1)))
    (fail 'shape-error "SAMPLING-RATE must be a positive integer"))
  (unless (or (null num-frames) (typep num-frames '(integer 1)))
    (fail 'shape-error "NUM-FRAMES must be a positive integer"))
  (unless (or (null fps) (and (realp fps) (plusp fps)))
    (fail 'shape-error "FPS must be a positive real number"))
  (let ((path (probe-file path)))
    (unless (and path (not (uiop:directory-pathname-p path)))
      (fail 'shape-error "Processor input file does not exist or is not a file"))
    (make-instance 'python-file-input :path (truename path) :media-type media-type
                   :sampling-rate sampling-rate :num-frames num-frames :fps fps)))

(defun make-chat-message (role content &rest fields)
  "Construct one JSON-compatible Hugging Face chat message."
  (unless (and (stringp role) (plusp (length role)) (evenp (length fields)))
    (fail 'shape-error "Chat role must be nonempty and extra fields must be key/value pairs"))
  (let ((message (make-hash-table :test 'equal)))
    (setf (gethash "role" message) role (gethash "content" message) content)
    (loop for (name value) on fields by #'cddr do
      (let ((key (etypecase name
                   (string name)
                   (symbol (substitute #\_ #\- (string-downcase name))))))
        (when (member key '("role" "content") :test #'equal)
          (fail 'shape-error "Chat extra fields cannot replace role or content"))
        (setf (gethash key message) value)))
    message))

(defun python-worker-alive-p (model)
  (and (not (python-model-disposed-p model))
       (uiop:process-alive-p (worker-process model))))

(defmethod model-backend ((model python-model))
  (declare (ignore model)) :python)

(defmethod model-capabilities ((model python-model))
  (let* ((information (python-worker-info model))
         (actual-device (gethash "actual_device" information))
         (tokenizer (gethash "tokenizer_class" information))
         (processor (gethash "processor_class" information))
         (operations '(:forward :training :generation :lora :rslora
                       :training-checkpoint :export :hub-push)))
    (when tokenizer (push :tokenization operations))
    (when processor (push :processor operations))
    (when (json-true-p (gethash "chat_template" information))
      (push :chat-template operations))
    (list :supported-p (python-worker-alive-p model)
          :loaded-p t
          :execution :python
          :backend :pytorch
          :model-type (gethash "model_type" (model-config model))
          :architecture (gethash "model_class" information)
          :source (python-model-source model)
          :revision (model-source-revision model)
          :supported-devices '(:cpu :gpu)
          :device (and actual-device (intern (string-upcase actual-device) :keyword))
          :requested-device
          (intern (string-upcase (gethash "requested_device" information)) :keyword)
          :device-api (and actual-device
                           (member actual-device '("mps" "cuda") :test #'equal)
                           (intern (string-upcase actual-device) :keyword))
          :operations (sort operations #'string< :key #'symbol-name)
          :checkpoint-dtype (gethash "dtype" (model-config model))
          :execution-dtype (gethash "model_dtype" information)
          :precision-policy :transformers
          :tokenizer tokenizer
          :processor processor
          :quantization (gethash "quantization_config" (model-config model))
          :compiled-p nil
          :mask-policy :model-defined
          :restrictions '("capabilities depend on the installed Transformers model class")
          :parameter-count nil
          :python-fallback-p nil
          :reason (unless (python-worker-alive-p model) "Python worker is not running"))))

(defun make-json-object (&rest pairs)
  (let ((object (make-hash-table :test 'equal)))
    (loop for (key value) on pairs by #'cddr do (setf (gethash key object) value))
    object))

(defun request-worker (model operation &rest pairs)
  (unless (python-worker-alive-p model)
    (fail 'backend-error "Python worker is not running"))
  (let* ((id (incf (worker-request-id model)))
         (request (apply #'make-json-object "id" id "op" operation pairs)))
    (handler-case
        (progn
          (yason:encode request (worker-input model))
          (terpri (worker-input model))
          (finish-output (worker-input model))
          (let* ((line (read-line (worker-output model)))
                 (response (yason:parse line :json-arrays-as-vectors t
                                             :json-booleans-as-symbols t)))
            (unless (eql id (gethash "id" response))
              (fail 'backend-error "Python worker protocol response ID mismatch"))
            (unless (json-true-p (gethash "ok" response))
              (fail 'compatibility-error "Python worker ~A: ~A"
                    (gethash "error_type" response) (gethash "message" response)))
            (gethash "result" response)))
      (end-of-file () (fail 'backend-error "Python worker exited during request ~A" operation))
      (stream-error (error) (fail 'backend-error "Python worker transport failed: ~A" error)))))

(defun array-as-json (array)
  (unless (and (arrayp array) (= 2 (array-rank array))
               (plusp (array-dimension array 0)) (plusp (array-dimension array 1)))
    (fail 'shape-error "Token IDs must be a nonempty rank-two Lisp array"))
  (let ((rows (make-array (array-dimension array 0))))
    (dotimes (row (length rows) rows)
      (let ((values (make-array (array-dimension array 1))))
        (dotimes (column (length values))
          (let ((value (aref array row column)))
            (unless (typep value '(integer 0))
              (fail 'shape-error "Token IDs and masks must be nonnegative integers"))
            (setf (aref values column) value)))
        (setf (aref rows row) values)))))

(defun response-tensor (response)
  (let* ((shape (coerce (gethash "shape" response) 'list))
         (data (gethash "data" response))
         (dtype-name (gethash "dtype" response "float32"))
         (dtype (cond ((equal dtype-name "int64") :int64)
                      ((equal dtype-name "bool") :bool)
                      ((equal dtype-name "float32") :float32)
                      (t (fail 'backend-error "Python worker returned unsupported dtype ~A"
                               dtype-name))))
         (size (reduce #'* shape :initial-value 1)))
    (unless (= size (length data))
      (fail 'backend-error "Python worker returned an invalid tensor payload"))
    (let ((array (make-array shape :element-type
                             (ecase dtype
                               (:float32 'single-float)
                               (:int64 '(signed-byte 64))
                               (:bool 'bit)))))
      (dotimes (index size)
        (setf (row-major-aref array index)
              (ecase dtype
                (:float32 (coerce (aref data index) 'single-float))
                (:int64 (aref data index))
                (:bool (if (json-true-p (aref data index)) 1 0)))))
      (make-instance 'python-tensor :shape shape :dtype dtype :array array))))

(defun python-array-spec (array)
  (let ((all-integers t) (all-reals t)
        (data (make-array (array-total-size array))))
    (dotimes (index (length data))
      (let ((value (row-major-aref array index)))
        (unless (integerp value) (setf all-integers nil))
        (unless (realp value) (setf all-reals nil))
        (setf (aref data index) value)))
    (unless all-reals
      (fail 'shape-error "Python tensor inputs must contain real numbers"))
    (make-json-object "kind" "tensor" "shape" (coerce (array-dimensions array) 'vector)
                      "dtype" (if all-integers "int64" "float32") "data" data)))

(defun binary-little-endian-p ()
  (cffi:with-foreign-object (value :uint16)
    (setf (cffi:mem-ref value :uint16) #x0102)
    (= (cffi:mem-aref value :uint8 0) 2)))

(defun python-binary-path (model)
  (unless (binary-little-endian-p)
    (fail 'compatibility-error "Binary worker transport requires a little-endian host"))
  (let ((directory (gethash "binary_directory" (python-worker-info model))))
    (unless (and (stringp directory) (uiop:directory-exists-p directory))
      (fail 'backend-error "Python worker has no binary transfer directory"))
    (merge-pathnames (format nil "lisp-~D-~D.bin" (worker-request-id model)
                             (random 1000000000))
                     (uiop:ensure-directory-pathname directory))))

(defun write-python-binary-array (model array)
  (let* ((all-integers (loop for index below (array-total-size array)
                             always (integerp (row-major-aref array index))))
         (all-reals (loop for index below (array-total-size array)
                          always (realp (row-major-aref array index))))
         (dtype (if all-integers "int64" "float32"))
         (path (python-binary-path model)))
    (unless all-reals (fail 'shape-error "Python tensor inputs must contain real numbers"))
    (with-open-file (stream path :direction :output :if-exists :error
                                 :element-type '(unsigned-byte 8))
      (if all-integers
          (dotimes (index (array-total-size array))
            (unless (typep (row-major-aref array index) '(signed-byte 64))
              (fail 'shape-error "Binary integer tensor values must fit int64"))
            (let ((bits (mod (row-major-aref array index) (ash 1 64))))
              (dotimes (byte 8) (write-byte (ldb (byte 8 (* byte 8)) bits) stream))))
          (cffi:with-foreign-object (cell :float)
            (dotimes (index (array-total-size array))
              (setf (cffi:mem-ref cell :float)
                    (coerce (row-major-aref array index) 'single-float))
              (dotimes (byte 4) (write-byte (cffi:mem-aref cell :uint8 byte) stream))))))
    (values (make-json-object "kind" "binary_tensor" "path" (namestring path)
                              "shape" (coerce (array-dimensions array) 'vector)
                              "dtype" dtype)
            path)))

(defun python-inputs-json (inputs &key binary-model)
  (unless (and (listp inputs) inputs)
    (fail 'shape-error "Python inputs must be a nonempty association list"))
  (let ((result (make-hash-table :test 'equal)) (files nil))
    (dolist (entry inputs (values result files))
      (unless (and (consp entry) (stringp (car entry)))
        (fail 'shape-error "Python input names must be unique strings"))
      (multiple-value-bind (value present) (gethash (car entry) result)
        (declare (ignore value))
        (when present (fail 'shape-error "Python input names must be unique strings")))
      (if (and (arrayp (cdr entry)) (not (stringp (cdr entry))))
          (if binary-model
              (multiple-value-bind (spec path)
                  (write-python-binary-array binary-model (cdr entry))
                (setf (gethash (car entry) result) spec)
                (push path files))
              (setf (gethash (car entry) result) (python-array-spec (cdr entry))))
          (setf (gethash (car entry) result) (cdr entry))))))

(defun read-python-binary-tensor (model response)
  (let* ((root (truename (uiop:ensure-directory-pathname
                          (gethash "binary_directory" (python-worker-info model)))))
         (path (probe-file (gethash "path" response)))
         (shape (coerce (gethash "shape" response) 'list))
         (dtype-name (gethash "dtype" response))
         (bytes-per-value (cond ((equal dtype-name "float32") 4)
                                ((equal dtype-name "int64") 8)
                                ((equal dtype-name "bool") 1)
                                (t (fail 'backend-error "Unsupported binary dtype ~A" dtype-name))))
         (size (reduce #'* shape :initial-value 1)))
    (unless (and path (equal (truename (uiop:pathname-directory-pathname path)) root))
      (fail 'backend-error "Worker returned a binary path outside its transfer directory"))
    (unwind-protect
         (with-open-file (stream path :element-type '(unsigned-byte 8))
           (unless (= (file-length stream) (* size bytes-per-value))
             (fail 'backend-error "Worker binary tensor byte length is invalid"))
           (let ((array (make-array shape :element-type
                                    (cond ((equal dtype-name "float32") 'single-float)
                                          ((equal dtype-name "int64") '(signed-byte 64))
                                          (t 'bit)))))
             (cffi:with-foreign-object (cell :float)
               (dotimes (index size)
                 (setf (row-major-aref array index)
                       (cond
                         ((equal dtype-name "float32")
                          (dotimes (byte 4)
                            (setf (cffi:mem-aref cell :uint8 byte) (read-byte stream)))
                          (cffi:mem-ref cell :float))
                         ((equal dtype-name "int64")
                          (let ((bits (loop for byte below 8
                                           sum (ash (read-byte stream) (* byte 8)))))
                            (if (logbitp 63 bits) (- bits (ash 1 64)) bits)))
                         (t (if (zerop (read-byte stream)) 0 1))))))
             (make-instance 'python-tensor :shape shape
                            :dtype (cond ((equal dtype-name "float32") :float32)
                                         ((equal dtype-name "int64") :int64) (t :bool))
                            :array array)))
      (when path (ignore-errors (delete-file path))))))

(defun processor-value-json (value)
  (typecase value
    (null nil)
    (python-file-input
     (make-json-object "kind" "file" "path" (namestring (python-file-path value))
                       "media_type" (string-downcase (python-file-media-type value))
                       "sampling_rate" (python-file-sampling-rate value)
                       "num_frames" (python-file-num-frames value)
                       "fps" (python-file-fps value)))
    (string value)
    (array
     (if (every #'realp value)
         (let ((spec (python-array-spec value)))
           (setf (gethash "kind" spec) "array") spec)
         (if (= 1 (array-rank value))
             (map 'vector #'processor-value-json value)
             (fail 'shape-error "Nonnumeric processor arrays must have rank one"))))
    (list (map 'vector #'processor-value-json value))
    (t value)))

(defun processor-inputs-json (inputs)
  (unless (listp inputs) (fail 'shape-error "Processor inputs must be an association list"))
  (let ((result (make-hash-table :test 'equal)))
    (dolist (entry inputs result)
      (unless (and (consp entry) (stringp (car entry)))
        (fail 'shape-error "Processor input names must be strings"))
      (multiple-value-bind (old present) (gethash (car entry) result)
        (declare (ignore old))
        (when present (fail 'shape-error "Processor input names must be unique")))
      (setf (gethash (car entry) result) (processor-value-json (cdr entry))))))

(defun response-tensors (response outputs)
  (loop for name in outputs collect
    (cons name (response-tensor (gethash name response)))))

(defmethod python-forward ((model python-model) inputs &key (outputs '("logits"))
                                                       (transport :json))
  "Call a Python AutoModel with named inputs; return named owned tensors."
  (unless (and (listp outputs) outputs (every #'stringp outputs))
    (fail 'shape-error "OUTPUTS must be a nonempty list of field names"))
  (unless (member transport '(:json :binary))
    (fail 'shape-error "TRANSPORT must be :JSON or :BINARY"))
  (multiple-value-bind (encoded files)
      (python-inputs-json inputs :binary-model (and (eq transport :binary) model))
    (unwind-protect
         (let ((response (request-worker model "python_forward"
                                         "inputs" encoded
                                         "outputs" (coerce outputs 'vector)
                                         "transport" (string-downcase transport))))
           (if (eq transport :binary)
               (loop for name in outputs collect
                 (cons name (read-python-binary-tensor model (gethash name response))))
               (response-tensors response outputs)))
      (dolist (path files)
        (when (probe-file path) (ignore-errors (delete-file path)))))))

(defmethod python-process ((model python-model) inputs
                           &key (options nil) (outputs '("input_ids" "attention_mask")))
  "Run the resident AutoProcessor and return selected prepared tensors."
  (let ((response (request-worker model "process"
                                  "inputs" (processor-inputs-json inputs)
                                  "options" (processor-inputs-json options)
                                  "outputs" (coerce outputs 'vector))))
    (response-tensors response outputs)))

(defmethod python-processor-forward ((model python-model) inputs
                                     &key (options nil) (model-inputs nil)
                                          (outputs '("logits")))
  "Prepare high-level inputs and invoke the resident AutoModel without tensor round trips."
  (let ((response (request-worker model "processor_forward"
                                  "inputs" (processor-inputs-json inputs)
                                  "options" (processor-inputs-json options)
                                  "model_inputs" (if model-inputs
                                                     (python-inputs-json model-inputs)
                                                     (make-json-object))
                                  "outputs" (coerce outputs 'vector))))
    (response-tensors response outputs)))

(defmethod python-generate ((model python-model) inputs
                            &key (options nil) (outputs '("sequences")) seed)
  "Run Transformers generation from named model inputs and return selected tensor fields."
  (let ((response (request-worker model "python_generate"
                                  "inputs" (python-inputs-json inputs)
                                  "options" (processor-inputs-json options)
                                  "outputs" (coerce outputs 'vector) "seed" seed)))
    (response-tensors response outputs)))

(defmethod python-processor-generate ((model python-model) inputs
                                      &key (processor-options nil) (options nil) (model-inputs nil)
                                           (outputs '("sequences")) seed)
  "Prepare high-level inputs and run Transformers generation inside the worker."
  (let ((response (request-worker model "processor_generate"
                                  "inputs" (processor-inputs-json inputs)
                                  "processor_options" (processor-inputs-json processor-options)
                                  "model_inputs" (if model-inputs
                                                     (python-inputs-json model-inputs)
                                                     (make-json-object))
                                  "generation_options" (processor-inputs-json options)
                                  "outputs" (coerce outputs 'vector) "seed" seed)))
    (response-tensors response outputs)))

(defmethod python-apply-chat-template
    ((model python-model) messages &key (add-generation-prompt nil)
                                       (continue-final-message nil) (tokenize nil)
                                       (options nil))
  "Apply the resident tokenizer's saved chat template and return text or token IDs."
  (unless (and (typep messages 'sequence) (plusp (length messages))
               (every #'hash-table-p messages))
    (fail 'shape-error "MESSAGES must be a nonempty sequence of chat-message objects"))
  (when (and add-generation-prompt continue-final-message)
    (fail 'shape-error "ADD-GENERATION-PROMPT and CONTINUE-FINAL-MESSAGE are exclusive"))
  (let ((response
          (request-worker model "apply_chat_template"
                          "messages" (map 'vector #'identity messages)
                          "add_generation_prompt"
                          (if add-generation-prompt 'yason:true 'yason:false)
                          "continue_final_message"
                          (if continue-final-message 'yason:true 'yason:false)
                          "tokenize" (if tokenize 'yason:true 'yason:false)
                          "options" (processor-inputs-json options))))
    (if tokenize (gethash "input_ids" response) (gethash "text" response))))

(defmethod tensor-array ((tensor python-tensor))
  (or (python-tensor-array tensor) (fail 'backend-error "Python tensor is disposed")))

(defmethod dispose ((tensor python-tensor))
  (setf (python-tensor-array tensor) nil))

(defmethod forward ((model python-model) inputs &key cache attention-mask token-type-ids)
  (when cache (fail 'compatibility-error "Python worker forward caches are not exposed; use GENERATE"))
  (let ((ids (array-as-json inputs))
        (mask (when attention-mask (array-as-json attention-mask)))
        (types (when token-type-ids (array-as-json token-type-ids))))
    (when (and mask (not (equal (array-dimensions inputs) (array-dimensions attention-mask))))
      (fail 'shape-error "Attention mask must match token IDs"))
    (when (and types (not (equal (array-dimensions inputs) (array-dimensions token-type-ids))))
      (fail 'shape-error "Token-type IDs must match token IDs"))
    (response-tensor
     (request-worker model "forward" "input_ids" ids "attention_mask" mask
                                    "token_type_ids" types))))

(defmethod encode-text ((model python-model) text &key (add-special-tokens t))
  (unless (stringp text) (fail 'shape-error "Text must be a string"))
  (request-worker model "encode" "text" text
                  "add_special_tokens" (if add-special-tokens 'yason:true 'yason:false)))

(defmethod decode-tokens ((model python-model) ids &key (skip-special-tokens nil))
  (unless (and (vectorp ids) (every (lambda (id) (typep id '(integer 0))) ids))
    (fail 'shape-error "Expected a vector of nonnegative token IDs"))
  (request-worker model "decode" "ids" ids
                  "skip_special_tokens" (if skip-special-tokens 'yason:true 'yason:false)))

(defmethod generate ((model python-model) prompt &key (max-new-tokens 20)
                                                (eos-token-id nil))
  (unless (typep max-new-tokens '(integer 0)) (fail 'shape-error "Invalid token limit"))
  (let ((ids (if (stringp prompt) (encode-text model prompt) prompt)))
    (unless (and (vectorp ids) (plusp (length ids))
                 (every (lambda (id) (typep id '(integer 0))) ids))
      (fail 'shape-error "Prompt must be a nonempty token vector or string"))
    (request-worker model "generate" "ids" ids "max_new_tokens" max-new-tokens
                    "eos_token_id" eos-token-id)))

(defmethod named-parameters ((model python-model))
  (declare (ignore model))
  (fail 'compatibility-error "Python worker parameters are intentionally process-resident"))

(defmethod make-cache ((model python-model) &key growth-step)
  (declare (ignore model growth-step))
  (fail 'compatibility-error "Python worker caches are internal to GENERATE"))

(defun python-export-destination (model destination)
  (let ((destination (uiop:ensure-directory-pathname destination)))
    (when (and (probe-file (python-model-source model))
               (uiop:directory-exists-p destination)
               (equal (truename (uiop:ensure-directory-pathname (python-model-source model)))
                      (truename destination)))
      (fail 'compatibility-error "Export to a separate directory to preserve the source checkpoint"))
    destination))

(defmethod save-pretrained ((model python-model) destination &key max-shard-size)
  (unless (or (null max-shard-size) (typep max-shard-size '(integer 1)))
    (fail 'shape-error "MAX-SHARD-SIZE must be NIL or a positive byte count"))
  (let ((destination (python-export-destination model destination)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (request-worker model "save" "destination" (namestring staging)
                       "max_shard_size" max-shard-size)))))

(defmethod push-to-hub ((model python-model) repo-id
                        &key (private nil) revision commit-message commit-description
                             (create-pr nil) model-card dry-run-directory)
  "Publish model, tokenizer, processor, and optional model card using inherited HF credentials."
  (unless (valid-hub-repo-id-p repo-id)
    (fail 'shape-error "Invalid Hugging Face model repository ID"))
  (request-worker model "push_to_hub" "repo_id" repo-id
                  "private" (if private 'yason:true 'yason:false)
                  "revision" revision "commit_message" commit-message
                  "commit_description" commit-description
                  "create_pr" (if create-pr 'yason:true 'yason:false)
                  "model_card" model-card
                  "dry_run_directory"
                  (when dry-run-directory
                    (namestring (python-export-destination model dry-run-directory)))))

(defun python-string-list (values name &key allow-nil)
  (unless (or (and allow-nil (null values))
              (and (listp values) values (every #'stringp values)
                   (= (length values) (length (remove-duplicates values :test #'equal)))))
    (fail 'shape-error "~A must be a nonempty list of unique strings" name))
  (and values (coerce values 'vector)))

(defmethod python-adapter-info ((model python-model) &key adapter-name)
  "Return active PEFT adapters, their standard configurations, and parameter counts."
  (unless (or (null adapter-name) (and (stringp adapter-name) (plusp (length adapter-name))))
    (fail 'shape-error "ADAPTER-NAME must be NIL or a nonempty string"))
  (request-worker model "adapter_info" "adapter_name" adapter-name))

(defmethod python-make-lora
    ((model python-model) &key (rank 8) (alpha 16) target-modules (dropout 0.0)
                               (rslora nil) (use-dora nil) (bias "none")
                               (task-type "CAUSAL_LM") modules-to-save
                               (adapter-name "default"))
  "Attach a standard PEFT LoRA/rsLoRA/DoRA adapter to the resident model."
  (unless (and (typep rank '(integer 1)) (realp alpha) (plusp alpha)
               (realp dropout) (<= 0 dropout) (< dropout 1)
               (member bias '("none" "all" "lora_only") :test #'equal)
               (stringp task-type) (plusp (length task-type))
               (stringp adapter-name) (plusp (length adapter-name)))
    (fail 'shape-error "Invalid worker LoRA configuration"))
  (request-worker model "make_lora" "rank" rank "alpha" alpha
                  "target_modules" (python-string-list target-modules "TARGET-MODULES" :allow-nil t)
                  "dropout" dropout "use_rslora" (if rslora 'yason:true 'yason:false)
                  "use_dora" (if use-dora 'yason:true 'yason:false) "bias" bias
                  "task_type" task-type
                  "modules_to_save" (python-string-list modules-to-save "MODULES-TO-SAVE"
                                                        :allow-nil t)
                  "adapter_name" adapter-name))

(defmethod python-load-adapter
    ((model python-model) source &key (adapter-name "default") (trainable nil)
                                revision (local-files-only nil))
  "Load a local or Hub PEFT adapter and make it active in the resident model."
  (unless (and (stringp adapter-name) (plusp (length adapter-name)))
    (fail 'shape-error "ADAPTER-NAME must be a nonempty string"))
  (request-worker model "load_adapter"
                  "source" (etypecase source
                             (pathname (namestring source))
                             (string source))
                  "adapter_name" adapter-name
                  "trainable" (if trainable 'yason:true 'yason:false)
                  "revision" revision
                  "local_files_only" (if local-files-only 'yason:true 'yason:false)))

(defmethod python-save-adapter ((model python-model) destination &key adapter-name)
  "Write standard PEFT configuration and safetensors for one or all adapters."
  (let ((destination (python-export-destination model destination)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (request-worker model "save_adapter" "destination" (namestring staging)
                       "adapter_name" adapter-name)))))

(defmethod python-merge-adapter ((model python-model) &key (safe-merge t) adapter-names)
  "Merge selected active LoRA adapters and replace the worker model with a plain model."
  (request-worker model "merge_adapter"
                  "safe_merge" (if safe-merge 'yason:true 'yason:false)
                  "adapter_names" (python-string-list adapter-names "ADAPTER-NAMES"
                                                      :allow-nil t)))

(defun terminate-worker (process input output)
  (ignore-errors (close input))
  (ignore-errors (close output))
  (when (uiop:process-alive-p process)
    (ignore-errors (uiop:terminate-process process)))
  (ignore-errors (uiop:wait-process process)))

(defmethod dispose ((model python-model))
  (unless (python-model-disposed-p model)
    (trivial-garbage:cancel-finalization model)
    (when (python-worker-alive-p model)
      (ignore-errors (request-worker model "close")))
    (setf (python-model-disposed-p model) t)
    ;; Closing stdin guarantees EOF even when CLOSE failed. Let Python release
    ;; Torch resources normally; forceful termination remains the finalizer path.
    (ignore-errors (close (worker-input model)))
    (ignore-errors (uiop:wait-process (worker-process model)))
    (ignore-errors (close (worker-output model)))))

(defmethod load-python-model (source &key (device :cpu) revision (task :causal-lm)
                                       (local-files-only nil) (trust-remote-code nil)
                                       (dtype "auto") python-executable
                                       (max-output-elements 2000000) auto-class python-threads
                                       (training-dtype "float32"))
  (unless (member device '(:cpu :gpu))
    (fail 'compatibility-error "Python worker device must be :CPU or :GPU"))
  (unless (typep max-output-elements '(integer 1 100000000))
    (fail 'compatibility-error "MAX-OUTPUT-ELEMENTS must be 1..100000000"))
  (unless (or (null python-threads) (typep python-threads '(integer 1 1024)))
    (fail 'compatibility-error "PYTHON-THREADS must be NIL or 1..1024"))
  (unless (member training-dtype '("float32" "float16" "bfloat16") :test #'equal)
    (fail 'compatibility-error "TRAINING-DTYPE must be float32, float16, or bfloat16"))
  (let* ((script (merge-pathnames "scripts/hf-worker.py"
                                  (asdf:system-source-directory "cl-transformer-blocks")))
         (process (uiop:launch-program
                   (list (or python-executable (default-python-executable)) "-u" (namestring script))
                   :input :stream :output :stream :error-output *error-output*))
         (input (uiop:process-info-input process))
         (output (uiop:process-info-output process))
         (model nil) (success nil))
    (unwind-protect
         (progn
           (setf model (make-instance 'python-model :process process :input input :output output
                                     :config (make-hash-table) :info (make-hash-table)
                                     :source source :revision revision))
           (let* ((task-name (string-downcase task))
                  (information
                    (request-worker model "load" "protocol_version" 1
                                    "source" (etypecase source
                                               (pathname (namestring source))
                                               (string source))
                                    "device" (string-downcase device) "revision" revision
                                    "task" task-name
                                    "auto_class" auto-class
                                    "num_threads" python-threads
                                    "training_dtype" training-dtype
                                    "local_files_only" (if local-files-only 'yason:true 'yason:false)
                                    "trust_remote_code" (if trust-remote-code 'yason:true 'yason:false)
                                    "dtype" dtype "max_output_elements" max-output-elements)))
             (unless (= 1 (gethash "protocol_version" information))
               (fail 'compatibility-error "Unsupported Python worker protocol"))
             (setf (slot-value model 'info) information
                   (slot-value model 'config) (gethash "config" information)))
           (let ((p process) (in input) (out output))
             (trivial-garbage:finalize model (lambda () (terminate-worker p in out))))
           (setf success t)
           model)
      (unless success
        (when model (setf (python-model-disposed-p model) t))
        (terminate-worker process input output)))))
