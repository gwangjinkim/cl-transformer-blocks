(in-package #:tb)
(defvar *next-optimizer-id* 0)
(defclass optimizer ()
  ((rate :initarg :rate :reader optimizer-rate)
   (decay :initarg :decay :reader optimizer-decay)
   (step :initform 0 :accessor optimizer-step)
   (state :initform (make-hash-table :test 'equal) :accessor optimizer-state)
   (owner :initform nil :accessor optimizer-owner)
   (version :initform nil :accessor optimizer-version)
   (remote-id :initform (incf *next-optimizer-id*) :reader optimizer-remote-id)
   (disposed :initform nil :accessor optimizer-disposed)))
(defclass sgd (optimizer) ((momentum :initarg :momentum :reader sgd-momentum)))
(defclass adamw (optimizer) ((beta1 :initarg :beta1 :reader adam-beta1)
                           (beta2 :initarg :beta2 :reader adam-beta2)
                           (epsilon :initarg :epsilon :reader adam-epsilon)))
(defun valid-real (value low high &optional open-low)
  (and (realp value) (if open-low (> value low) (>= value low)) (< value high)))
(defun validate-optimizer-options (rate decay)
  (unless (and (valid-real rate 0 most-positive-single-float t)
               (valid-real decay 0 most-positive-single-float))
    (fail 'shape-error "Invalid learning rate or weight decay")))
(defun make-sgd (&key (learning-rate 0.01) (momentum 0.0) (weight-decay 0.0))
  (validate-optimizer-options learning-rate weight-decay)
  (unless (valid-real momentum 0 1) (fail 'shape-error "Momentum must be in [0,1)"))
  (make-instance 'sgd :rate learning-rate :decay weight-decay :momentum momentum))
(defun make-adamw (&key (learning-rate 0.001) (beta1 0.9) (beta2 0.999) (epsilon 1e-8) (weight-decay 0.01))
  (validate-optimizer-options learning-rate weight-decay)
  (unless (and (valid-real beta1 0 1) (valid-real beta2 0 1) (valid-real epsilon 0 1 t))
    (fail 'shape-error "Invalid AdamW beta/epsilon"))
  (make-instance 'adamw :rate learning-rate :decay weight-decay :beta1 beta1 :beta2 beta2 :epsilon epsilon))
(defun dispose-state (state)
  (maphash (lambda (k values) (declare (ignore k)) (mapc #'dispose values)) state))
(defmethod dispose ((optimizer optimizer))
  (when (and (typep (optimizer-owner optimizer) 'python-model)
             (python-worker-alive-p (optimizer-owner optimizer)))
    (ignore-errors
      (request-worker (optimizer-owner optimizer) "drop_optimizer"
                      "optimizer_id" (optimizer-remote-id optimizer))))
  (dispose-state (optimizer-state optimizer)) (clrhash (optimizer-state optimizer))
  (setf (optimizer-disposed optimizer) t (optimizer-owner optimizer) nil))
(defgeneric propose-update (optimizer parameter gradient state step))
(defmethod propose-update ((optimizer sgd) parameter gradient state step)
  (declare (ignore step))
  (let* ((g (if (zerop (optimizer-decay optimizer)) gradient
                (add gradient (multiply (scalar (optimizer-decay optimizer)) parameter))))
         (velocity (if (and state (plusp (sgd-momentum optimizer)))
                       (add g (multiply (scalar (sgd-momentum optimizer)) (first state))) g)))
    (values (subtract parameter (multiply (scalar (optimizer-rate optimizer)) velocity))
            (when (plusp (sgd-momentum optimizer)) (list velocity)))))
(defmethod propose-update ((optimizer adamw) parameter gradient state step)
  (let* ((b1 (adam-beta1 optimizer)) (b2 (adam-beta2 optimizer))
         (m0 (if state (first state) (scalar 0.0))) (v0 (if state (second state) (scalar 0.0)))
         (m (add (multiply (scalar b1) m0) (multiply (scalar (- 1 b1)) gradient)))
         (v (add (multiply (scalar b2) v0) (multiply (scalar (- 1 b2)) (multiply gradient gradient))))
         (mhat (divide m (scalar (- 1 (expt b1 step)))))
         (vhat (divide v (scalar (- 1 (expt b2 step)))))
         (delta (multiply (scalar (optimizer-rate optimizer))
                          (divide mhat (add (tensor-sqrt vhat) (scalar (adam-epsilon optimizer)))))))
    (values (subtract (multiply (scalar (- 1 (* (optimizer-rate optimizer) (optimizer-decay optimizer)))) parameter) delta)
            (list m v))))
(defun gradient-norm (gradients)
  (let ((sum (scalar 0.0)))
    (dolist (entry gradients) (setf sum (add sum (tensor-sum (multiply (cdr entry) (cdr entry))))))
    (let ((norm (require-finite (tensor-sqrt sum)))) (row-major-aref (tensor-array norm) 0))))
(defmethod train-step ((model model) (optimizer optimizer) ids
                       &key labels attention-mask token-type-ids max-grad-norm)
  "Compute a masked loss and atomically update parameters and optimizer state.
A bound optimizer rejects external model/adapter mutations; create a new optimizer then."
  (when (or (optimizer-disposed optimizer)
            (and (optimizer-owner optimizer)
                 (or (not (eq model (optimizer-owner optimizer)))
                     (/= (model-version model) (optimizer-version optimizer)))))
    (fail 'compatibility-error "Optimizer is disposed or belongs to a different model version"))
  (when (and max-grad-norm (not (valid-real max-grad-norm 0 most-positive-single-float t)))
    (fail 'shape-error "Gradient norm limit must be positive"))
  (multiple-value-bind (loss gradients)
      (loss-and-gradients model ids :labels labels :attention-mask attention-mask
                                   :token-type-ids token-type-ids)
    (let ((*backend* (model-backend model)) (updated (make-hash-table :test 'equal))
          (new-state (make-hash-table :test 'equal)) (committed nil)
          (table (if (model-adapter model) (adapter-parameters (model-adapter model)) (parameters model))))
      (unwind-protect
           (progn
             (with-tensor-scope
               (let ((scale (if max-grad-norm (min 1.0 (/ max-grad-norm (+ (gradient-norm gradients) 1e-6))) 1.0)))
                 (dolist (entry gradients)
                   (multiple-value-bind (next state)
                       (propose-update optimizer (gethash (car entry) table)
                                       (multiply (scalar scale) (cdr entry))
                                       (gethash (car entry) (optimizer-state optimizer)) (1+ (optimizer-step optimizer)))
                     (require-finite next)
                     (mapc #'require-finite state)
                     (setf (gethash (car entry) updated) (retain-tensor next)
                           (gethash (car entry) new-state) (mapcar #'retain-tensor state))))))
             (dolist (entry gradients)
               (dispose (gethash (car entry) table))
               (setf (gethash (car entry) table) (gethash (car entry) updated)))
             (dispose-state (optimizer-state optimizer))
             (setf (optimizer-state optimizer) new-state (optimizer-owner optimizer) model committed t)
             (unless (model-adapter model) (setf (base-updated-p model) t))
             (incf (optimizer-step optimizer)) (incf (model-version model))
             (setf (optimizer-version optimizer) (model-version model)) loss)
        (mapc (lambda (entry) (dispose (cdr entry))) gradients)
        (unless committed
          (maphash (lambda (k v) (declare (ignore k)) (dispose v)) updated)
          (dispose-state new-state))))))

(defgeneric optimizer-description (optimizer))
(defmethod optimizer-description ((optimizer sgd))
  (make-json-object "algorithm" "sgd" "learning_rate" (optimizer-rate optimizer)
                    "momentum" (sgd-momentum optimizer)
                    "weight_decay" (optimizer-decay optimizer)))
(defmethod optimizer-description ((optimizer adamw))
  (make-json-object "algorithm" "adamw" "learning_rate" (optimizer-rate optimizer)
                    "beta1" (adam-beta1 optimizer) "beta2" (adam-beta2 optimizer)
                    "epsilon" (adam-epsilon optimizer)
                    "weight_decay" (optimizer-decay optimizer)))

(defun json-value-equal-p (left right)
  (cond
    ((and (numberp left) (numberp right)) (= left right))
    ((and (hash-table-p left) (hash-table-p right))
     (and (= (hash-table-count left) (hash-table-count right))
          (loop for key being the hash-keys of left
                always (multiple-value-bind (value present) (gethash key right)
                         (and present (json-value-equal-p (gethash key left) value))))))
    ((and (vectorp left) (vectorp right))
     (and (= (length left) (length right))
          (loop for a across left for b across right always (json-value-equal-p a b))))
    (t (equal left right))))

(defgeneric optimizer-state-labels (optimizer))
(defmethod optimizer-state-labels ((optimizer sgd))
  (if (plusp (sgd-momentum optimizer)) '("momentum") nil))
(defmethod optimizer-state-labels ((optimizer adamw))
  '("first_moment" "second_moment"))

(defun optimizer-state-key (label parameter-name)
  (format nil "optimizer.~A.~A" label parameter-name))

(defun native-training-kind (model)
  (if (model-adapter model) "adapter" "full"))

(defun native-training-source-matches-p (model kind directory)
  (let ((source (if (equal kind "adapter")
                    (and (model-adapter model) (adapter-directory (model-adapter model)))
                    (model-directory model))))
    (and source (probe-file source)
         (equal (truename source) (truename directory)))))

(defmethod save-training-checkpoint ((model model) (optimizer optimizer) destination)
  "Save standard model/adapter artifacts plus versioned native optimizer state."
  (when (or (optimizer-disposed optimizer)
            (not (eq (optimizer-owner optimizer) model))
            (/= (optimizer-version optimizer) (model-version model))
            (zerop (optimizer-step optimizer)))
    (fail 'compatibility-error "Checkpoint requires an active optimizer bound to this model version"))
  (let* ((destination (uiop:ensure-directory-pathname destination))
         (kind (native-training-kind model))
         (parameters (trainable-parameters model))
         (labels (optimizer-state-labels optimizer))
         (state-tensors nil))
    (dolist (entry parameters)
      (let ((state (gethash (car entry) (optimizer-state optimizer))))
        (unless (= (length state) (length labels))
          (fail 'compatibility-error "Optimizer state is incomplete for ~A" (car entry)))
        (loop for label in labels for tensor in state do
          (push (cons (optimizer-state-key label (car entry)) tensor) state-tensors))))
    (setf destination
          (if (equal kind "adapter")
              (native-adapter-export-destination model destination)
              (native-model-export-destination model destination)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (if (equal kind "adapter")
           (write-native-adapter-directory model staging)
           (write-native-model-directory model staging nil))
       (when state-tensors
         (save-weights (nreverse state-tensors)
                       (merge-pathnames "optimizer.safetensors" staging)))
       (let ((manifest (make-hash-table :test 'equal)))
         (setf (gethash "format_version" manifest) 1
               (gethash "implementation" manifest) "cl-transformer-blocks/native-mlx"
               (gethash "model_kind" manifest) kind
               (gethash "model_type" manifest) (gethash "model_type" (model-config model))
               (gethash "step" manifest) (optimizer-step optimizer)
               (gethash "optimizer" manifest) (optimizer-description optimizer)
               (gethash "state_slots" manifest) (coerce labels 'vector)
               (gethash "state_file" manifest) (and state-tensors "optimizer.safetensors")
               (gethash "trainable_parameters" manifest)
               (coerce (mapcar #'car parameters) 'vector)
               (gethash "rng_state" manifest) "none-zero-dropout-native-training")
         (write-json manifest (merge-pathnames "training_state.json" staging)))))))

(defmethod restore-training-checkpoint ((model model) (optimizer optimizer) source
                                        &key (restore-rng t))
  "Restore native optimizer moments and step count into a newly loaded model/optimizer pair."
  (declare (ignore restore-rng))
  (when (or (optimizer-disposed optimizer) (optimizer-owner optimizer)
            (plusp (optimizer-step optimizer))
            (plusp (hash-table-count (optimizer-state optimizer))))
    (fail 'compatibility-error "Restore requires a fresh unbound optimizer"))
  (let* ((source (uiop:ensure-directory-pathname source))
         (manifest-path (merge-pathnames "training_state.json" source)))
    (unless (probe-file manifest-path)
      (fail 'compatibility-error "Native training checkpoint manifest is missing"))
    (let ((manifest (read-json manifest-path)))
      (unless (hash-table-p manifest)
        (fail 'compatibility-error "Native training checkpoint manifest must be an object"))
      (let* ((kind (gethash "model_kind" manifest))
             (parameters (trainable-parameters model))
             (names (mapcar #'car parameters))
             (stored-names (gethash "trainable_parameters" manifest))
             (labels (optimizer-state-labels optimizer))
             (stored-labels (gethash "state_slots" manifest)))
        (unless (and (vectorp stored-names) (vectorp stored-labels)
                     (= (gethash "format_version" manifest 0) 1)
                     (equal (gethash "implementation" manifest)
                            "cl-transformer-blocks/native-mlx")
                     (member kind '("full" "adapter") :test #'equal)
                     (equal kind (native-training-kind model))
                     (equal (gethash "model_type" manifest)
                            (gethash "model_type" (model-config model)))
                     (typep (gethash "step" manifest) '(integer 1))
                     (json-value-equal-p (gethash "optimizer" manifest)
                                         (optimizer-description optimizer))
                     (equal names (coerce stored-names 'list))
                     (equal labels (coerce stored-labels 'list))
                     (= (model-version model) (if (equal kind "adapter") 1 0))
                     (native-training-source-matches-p model kind source))
          (fail 'compatibility-error "Training checkpoint does not match the model or optimizer"))
        (let ((loaded (make-hash-table :test 'equal))
              (new-state (make-hash-table :test 'equal))
              (success nil))
          (unwind-protect
               (progn
                 (if labels
                     (let ((state-file (gethash "state_file" manifest)))
                       (unless (and (equal state-file "optimizer.safetensors")
                                    (probe-file (merge-pathnames state-file source)))
                         (fail 'compatibility-error "Native optimizer state file is missing"))
                       (load-weights (merge-pathnames state-file source)
                                     (model-backend model) loaded))
                     (when (gethash "state_file" manifest)
                       (fail 'compatibility-error "Stateless optimizer checkpoint names a state file")))
                 (unless (= (hash-table-count loaded) (* (length labels) (length parameters)))
                   (fail 'compatibility-error "Native optimizer tensor coverage mismatch"))
                 (dolist (entry parameters)
                   (let ((state
                           (loop for label in labels
                                 for key = (optimizer-state-key label (car entry))
                                 for tensor = (gethash key loaded)
                                 do (unless (and tensor (equal (tensor-shape tensor)
                                                              (tensor-shape (cdr entry))))
                                      (fail 'shape-error "Missing or misshaped optimizer tensor: ~A" key))
                                    (require-finite tensor)
                                 collect tensor)))
                     (setf (gethash (car entry) new-state) state)))
                 (setf (optimizer-state optimizer) new-state
                       (optimizer-owner optimizer) model
                       (optimizer-version optimizer) (model-version model)
                       (optimizer-step optimizer) (gethash "step" manifest)
                       success t)
                 (gethash "step" manifest))
            (if success
                (clrhash loaded)
                (maphash (lambda (key tensor) (declare (ignore key)) (dispose tensor))
                         loaded))))))))

(defmethod python-train-step ((model python-model) (optimizer optimizer) inputs
                              &key max-grad-norm)
  "Update a resident Python model using its scalar loss; return pre-update loss."
  (when (or (optimizer-disposed optimizer)
            (and (optimizer-owner optimizer) (not (eq model (optimizer-owner optimizer)))))
    (fail 'compatibility-error "Optimizer is disposed or belongs to another model"))
  (let ((response
          (request-worker model "train_step" "inputs" (python-inputs-json inputs)
                          "optimizer_id" (optimizer-remote-id optimizer)
                          "optimizer" (optimizer-description optimizer)
                          "max_grad_norm" max-grad-norm)))
    (setf (optimizer-owner optimizer) model)
    (incf (optimizer-step optimizer))
    (gethash "loss" response)))

(defmethod python-train-microbatches ((model python-model) (optimizer optimizer) microbatches
                                      &key max-grad-norm)
  "Average gradients from nonempty named-input microbatches and perform one optimizer step."
  (unless (and (listp microbatches) microbatches)
    (fail 'shape-error "MICROBATCHES must be a nonempty list"))
  (when (or (optimizer-disposed optimizer)
            (and (optimizer-owner optimizer) (not (eq model (optimizer-owner optimizer)))))
    (fail 'compatibility-error "Optimizer is disposed or belongs to another model"))
  (let ((response
          (request-worker model "train_microbatches"
                          "microbatches" (map 'vector #'python-inputs-json microbatches)
                          "optimizer_id" (optimizer-remote-id optimizer)
                          "optimizer" (optimizer-description optimizer)
                          "max_grad_norm" max-grad-norm)))
    (setf (optimizer-owner optimizer) model)
    (incf (optimizer-step optimizer))
    (gethash "loss" response)))

(defmethod configure-python-scheduler ((model python-model) (optimizer optimizer) kind
                                       &key (warmup-steps 0) total-steps
                                            (scheduler-options nil))
  "Attach one Transformers learning-rate scheduler before the optimizer's first step."
  (unless (and (typep warmup-steps '(integer 0))
               (or (null total-steps) (typep total-steps '(integer 1))))
    (fail 'shape-error "Invalid scheduler step counts"))
  (unless (or (keywordp kind) (stringp kind))
    (fail 'shape-error "Scheduler kind must be a keyword or string"))
  (let ((response
          (request-worker model "configure_scheduler"
                          "optimizer_id" (optimizer-remote-id optimizer)
                          "optimizer" (optimizer-description optimizer)
                          "scheduler"
                          (make-json-object
                           "name" (string-downcase kind) "warmup_steps" warmup-steps
                           "total_steps" total-steps
                           "options" (processor-inputs-json scheduler-options)))))
    (setf (optimizer-owner optimizer) model)
    response))

(defmethod python-optimizer-info ((model python-model) (optimizer optimizer))
  (unless (eq (optimizer-owner optimizer) model)
    (fail 'compatibility-error "Optimizer does not belong to this model"))
  (request-worker model "optimizer_info" "optimizer_id" (optimizer-remote-id optimizer)))

(defmethod train-step ((model python-model) (optimizer optimizer) ids
                       &key labels attention-mask token-type-ids max-grad-norm)
  (unless (and (arrayp ids) (= 2 (array-rank ids)))
    (fail 'shape-error "Token IDs must be a rank-two Lisp array"))
  (dolist (value (list labels attention-mask token-type-ids))
    (when (and value (not (and (arrayp value)
                               (equal (array-dimensions value) (array-dimensions ids)))))
      (fail 'shape-error "Labels, attention mask, and token-type IDs must match token IDs")))
  (let ((targets (let* ((source (or labels ids))
                        (copy (make-array (array-dimensions source))))
                   (dotimes (index (array-total-size source) copy)
                     (setf (row-major-aref copy index)
                           (if (and attention-mask
                                    (zerop (row-major-aref attention-mask index)))
                               -100 (row-major-aref source index))))))
        (inputs `(("input_ids" . ,ids))))
    (when attention-mask (setf inputs (append inputs `(("attention_mask" . ,attention-mask)))))
    (when token-type-ids (setf inputs (append inputs `(("token_type_ids" . ,token-type-ids)))))
    (setf inputs (append inputs `(("labels" . ,targets))))
    (python-train-step model optimizer inputs :max-grad-norm max-grad-norm)))

(defmethod python-processor-train-step
    ((model python-model) (optimizer optimizer) inputs
     &key (options nil) (model-inputs nil) max-grad-norm)
  "Prepare high-level inputs and update a resident Python model."
  (when (or (optimizer-disposed optimizer)
            (and (optimizer-owner optimizer) (not (eq model (optimizer-owner optimizer)))))
    (fail 'compatibility-error "Optimizer is disposed or belongs to another model"))
  (let ((response
          (request-worker model "processor_train_step"
                          "inputs" (processor-inputs-json inputs)
                          "options" (processor-inputs-json options)
                          "model_inputs" (if model-inputs
                                             (python-inputs-json model-inputs)
                                             (make-json-object))
                          "optimizer_id" (optimizer-remote-id optimizer)
                          "optimizer" (optimizer-description optimizer)
                          "max_grad_norm" max-grad-norm)))
    (setf (optimizer-owner optimizer) model)
    (incf (optimizer-step optimizer))
    (gethash "loss" response)))

(defmethod save-training-checkpoint ((model python-model) (optimizer optimizer) destination)
  "Save standard model artifacts plus one resumable worker optimizer and Torch RNG state."
  (unless (eq (optimizer-owner optimizer) model)
    (fail 'compatibility-error "Optimizer has not successfully updated this model"))
  (let ((destination (python-export-destination model destination)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (request-worker model "save_training_checkpoint"
                       "optimizer_id" (optimizer-remote-id optimizer)
                       "destination" (namestring staging))))))

(defmethod restore-training-checkpoint ((model python-model) (optimizer optimizer) source
                                        &key (restore-rng t))
  "Restore an optimizer and optional Torch RNG state; return its completed step count."
  (when (optimizer-owner optimizer)
    (fail 'compatibility-error "Restore requires a new unbound optimizer"))
  (let ((response
          (request-worker model "restore_training_checkpoint"
                          "optimizer_id" (optimizer-remote-id optimizer)
                          "optimizer" (optimizer-description optimizer)
                          "source" (namestring (uiop:ensure-directory-pathname source))
                          "restore_rng" (if restore-rng 'yason:true 'yason:false))))
    (setf (optimizer-owner optimizer) model
          (optimizer-step optimizer) (gethash "step" response))
    (optimizer-step optimizer)))
