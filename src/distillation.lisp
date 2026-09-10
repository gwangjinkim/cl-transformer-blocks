(in-package #:tb)

(define-native "tb_distillation_loss" :pointer
  (context :pointer) (student :pointer) (teacher :pointer) (temperature :float))
(define-native "tb_distillation_topk_target" :int
  (context :pointer) (teacher :pointer) (top-k :int) (temperature :float)
  (top-log-probs :pointer) (indices :pointer) (tail-log-prob :pointer))
(define-native "tb_distillation_topk_loss" :pointer
  (context :pointer) (student :pointer) (top-log-probs :pointer)
  (indices :pointer) (tail-log-prob :pointer) (temperature :float))

(defmethod architecture-capabilities :around (architecture config directory)
  (let ((result (call-next-method)))
    (when (member architecture '(llama-model composed-causal-lm))
      (push :distillation (getf result :operations))
      (push :distillation-targets (getf result :operations))
      (push :distillation-datasets (getf result :operations))
      (push "distillation requires a native Llama/composed peer or reusable target with matching token-ID meanings"
            (getf result :restrictions)))
    result))

(defclass distillation-batch ()
  ((backend :initarg :backend :reader distillation-batch-backend)
   (representation :initarg :representation :initform :dense
                   :reader distillation-batch-representation)
   (logits :initarg :logits :initform nil :reader distillation-batch-logits)
   (top-log-probs :initarg :top-log-probs :initform nil
                  :reader distillation-batch-top-log-probs)
   (top-indices :initarg :top-indices :initform nil
                :reader distillation-batch-top-indices)
   (tail-log-prob :initarg :tail-log-prob :initform nil
                  :reader distillation-batch-tail-log-prob)
   (top-k :initarg :top-k :initform nil :reader distillation-batch-top-k)
   (temperature :initarg :temperature :initform nil
                :reader distillation-batch-temperature)
   (ids :initarg :ids :reader distillation-batch-ids)
   (labels :initarg :labels :reader distillation-batch-labels)
   (mask :initarg :mask :reader distillation-batch-mask)
   (vocab-size :initarg :vocab-size :reader distillation-batch-vocab-size)
   (storage-dtype :initarg :storage-dtype :initform :float32
                  :reader distillation-batch-storage-dtype)
   (teacher-source :initarg :teacher-source :reader distillation-batch-teacher-source)
   (teacher-revision :initarg :teacher-revision :reader distillation-batch-teacher-revision)))

(defmethod dispose ((batch distillation-batch))
  (dolist (tensor (list (distillation-batch-logits batch)
                        (distillation-batch-top-log-probs batch)
                        (distillation-batch-top-indices batch)
                        (distillation-batch-tail-log-prob batch)))
    (dispose tensor))
  (dispose (distillation-batch-backend batch)))

(defun copy-array-values (array)
  (let ((copy (make-array (array-dimensions array))))
    (dotimes (index (array-total-size array) copy)
      (setf (row-major-aref copy index) (row-major-aref array index)))))

(defun validate-distillation-model (model)
  (unless (and (typep model 'model)
               (member (class-of model) (list (find-class 'llama-model)
                                               (find-class 'composed-causal-lm))))
    (fail 'compatibility-error "Distillation requires an exact native Llama or composed model"))
  (pointer (model-backend model))
  (validate-training-config model)
  model)

(defun make-distillation-batch (teacher ids &key labels attention-mask)
  "Materialize selected FP32 teacher logits for reuse after TEACHER is disposed.
The result owns an independent backend and copies the effective IDs, labels, and mask.
SAVE-DISTILLATION-BATCH publishes the portable FP32 or FP16 safetensors/JSON artifact."
  (validate-distillation-model teacher)
  (multiple-value-bind (inputs padding targets selected)
      (prepare-training-batch teacher ids labels attention-mask)
    (declare (ignore targets))
    (let* ((*backend* (model-backend teacher))
           (backend (make-backend :device (backend-device *backend*)))
           (owned nil) (success nil))
      (unwind-protect
           (with-tensor-scope
             (with-resource (logits (let ((*parameter-overrides* nil))
                                     (forward teacher inputs :attention-mask padding)))
               (let* ((flat (reshape logits (list -1 (config teacher "vocab_size"))))
                      (reference (require-finite
                                  (take-indices flat (tensor-from-array *backend* selected :dtype :int32)))))
                 (checked-status (%tb-tensor-eval (pointer reference)))
                 (let ((*scoped-p* nil))
                   (setf owned (wrap-tensor (%tb-tensor-retain (pointer reference)) backend)))
                 (setf success t)
                 (make-instance 'distillation-batch :backend backend :logits owned
                                :ids (copy-array-values ids)
                                :labels (copy-array-values (or labels ids))
                                :mask (and attention-mask (copy-array-values attention-mask))
                                :vocab-size (config teacher "vocab_size")
                                :teacher-source (model-source-id teacher)
                                :teacher-revision (model-source-revision teacher)))))
        (unless success (dispose owned) (dispose backend))))))

(defun validate-top-k-distillation-options (vocab-size top-k temperature)
  (unless (and (typep top-k `(integer 1 (,vocab-size)))
               (valid-real temperature 0 most-positive-single-float t))
    (fail 'shape-error
          "Top-k distillation requires 0 < TOP-K < vocabulary and a positive finite temperature")))

(defun make-top-k-distillation-batch
    (teacher ids &key labels attention-mask top-k (temperature 1.0))
  "Materialize a fixed-temperature teacher distribution over TOP-K tokens plus one tail.
The result snapshots selection arrays and owns native tensors independent of TEACHER."
  (validate-distillation-model teacher)
  (validate-top-k-distillation-options (config teacher "vocab_size") top-k temperature)
  (multiple-value-bind (inputs padding targets selected)
      (prepare-training-batch teacher ids labels attention-mask)
    (declare (ignore targets))
    (let* ((*backend* (model-backend teacher))
           (backend (make-backend :device (backend-device *backend*)))
           (top-log-probs nil) (top-indices nil) (tail-log-prob nil) (success nil))
      (unwind-protect
           (with-tensor-scope
             (with-resource (logits (let ((*parameter-overrides* nil))
                                     (forward teacher inputs :attention-mask padding)))
               (let* ((flat (reshape logits (list -1 (config teacher "vocab_size"))))
                      (reference (require-finite
                                  (take-indices
                                   flat (tensor-from-array *backend* selected :dtype :int32)))))
                 (cffi:with-foreign-objects
                     ((top-output :pointer) (index-output :pointer) (tail-output :pointer))
                   (setf (cffi:mem-ref top-output :pointer) (cffi:null-pointer)
                         (cffi:mem-ref index-output :pointer) (cffi:null-pointer)
                         (cffi:mem-ref tail-output :pointer) (cffi:null-pointer))
                   (checked-status
                    (%tb-distillation-topk-target
                     (pointer *backend*) (pointer reference) top-k (float temperature 1.0)
                     top-output index-output tail-output))
                   (let ((*scoped-p* nil))
                     (setf top-log-probs
                           (wrap-tensor (cffi:mem-ref top-output :pointer) backend)
                           top-indices
                           (wrap-tensor (cffi:mem-ref index-output :pointer) backend)
                           tail-log-prob
                           (wrap-tensor (cffi:mem-ref tail-output :pointer) backend)))
                   (dolist (tensor (list top-log-probs top-indices tail-log-prob))
                     (checked-status (%tb-tensor-eval (pointer tensor))))
                   (setf success t)
                   (make-instance
                    'distillation-batch :backend backend :representation :top-k
                    :top-log-probs top-log-probs :top-indices top-indices
                    :tail-log-prob tail-log-prob :top-k top-k
                    :temperature (float temperature 1.0)
                    :ids (copy-array-values ids) :labels (copy-array-values (or labels ids))
                    :mask (and attention-mask (copy-array-values attention-mask))
                    :vocab-size (config teacher "vocab_size")
                    :teacher-source (model-source-id teacher)
                    :teacher-revision (model-source-revision teacher))))))
        (unless success
          (dispose top-log-probs) (dispose top-indices) (dispose tail-log-prob)
          (dispose backend))))))

(defun validate-distillation-options (temperature hard-weight)
  (unless (and (valid-real temperature 0 most-positive-single-float t)
               (realp hard-weight) (<= 0 hard-weight 1))
    (fail 'shape-error "Temperature must be positive and finite; hard weight must be in [0,1]")))

(defun call-with-distillation-reference
    (student reference ids labels mask temperature hard-weight function)
  (validate-distillation-model student)
  (validate-distillation-options temperature hard-weight)
  (multiple-value-bind (inputs padding targets selected)
      (prepare-training-batch student ids labels mask)
    (declare (ignore inputs padding targets))
    (unless (equal (tensor-shape reference) (list (length selected) (config student "vocab_size")))
      (fail 'compatibility-error "Teacher targets do not match selected positions and student vocabulary"))
    (unless (eq (backend-device (tensor-backend reference))
                (backend-device (model-backend student)))
      (fail 'compatibility-error "Teacher targets and student require the same device"))
    (let ((*training-loss-function*
            (lambda (predictions supervised)
              (let ((soft (unless (= hard-weight 1)
                            (wrap-tensor (%tb-distillation-loss
                                          (pointer *backend*) (pointer predictions) (pointer reference)
                                          (float temperature 1.0)) *backend*))))
                (cond ((zerop hard-weight) soft)
                      ((= hard-weight 1) (cross-entropy predictions supervised))
                      (t (add (multiply (scalar (- 1 hard-weight)) soft)
                              (multiply (scalar hard-weight) (cross-entropy predictions supervised)))))))))
      (let ((*scoped-p* nil)) (funcall function)))))

(defun call-with-top-k-distillation-reference
    (student top-log-probs indices tail-log-prob ids labels mask
     target-temperature temperature hard-weight function)
  (validate-distillation-model student)
  (validate-distillation-options temperature hard-weight)
  (unless (= (float temperature 1.0) target-temperature)
    (fail 'compatibility-error
          "Top-k targets were materialized at temperature ~S, not ~S"
          target-temperature temperature))
  (multiple-value-bind (inputs padding targets selected)
      (prepare-training-batch student ids labels mask)
    (declare (ignore inputs padding targets))
    (let ((rows (length selected)) (vocab-size (config student "vocab_size")))
      (unless (and (equal (tensor-shape top-log-probs)
                          (tensor-shape indices))
                   (equal (tensor-shape tail-log-prob) (list rows 1))
                   (= (first (tensor-shape top-log-probs)) rows)
                   (< 0 (second (tensor-shape top-log-probs)) vocab-size))
        (fail 'compatibility-error
              "Top-k teacher targets do not match selected positions and student vocabulary")))
    (unless (every (lambda (tensor)
                     (eq (backend-device (tensor-backend tensor))
                         (backend-device (model-backend student))))
                   (list top-log-probs indices tail-log-prob))
      (fail 'compatibility-error "Teacher targets and student require the same device"))
    (let ((*training-loss-function*
            (lambda (predictions supervised)
              (let ((soft
                      (unless (= hard-weight 1)
                        (wrap-tensor
                         (%tb-distillation-topk-loss
                          (pointer *backend*) (pointer predictions)
                          (pointer top-log-probs) (pointer indices)
                          (pointer tail-log-prob) (float temperature 1.0))
                         *backend*))))
                (cond ((zerop hard-weight) soft)
                      ((= hard-weight 1) (cross-entropy predictions supervised))
                      (t (add (multiply (scalar (- 1 hard-weight)) soft)
                              (multiply (scalar hard-weight)
                                        (cross-entropy predictions supervised)))))))))
      (let ((*scoped-p* nil)) (funcall function)))))

(defun call-with-distillation-batch-reference
    (student batch temperature hard-weight function)
  (ecase (distillation-batch-representation batch)
    (:dense
     (call-with-distillation-reference
      student (distillation-batch-logits batch) (distillation-batch-ids batch)
      (distillation-batch-labels batch) (distillation-batch-mask batch)
      temperature hard-weight function))
    (:top-k
     (call-with-top-k-distillation-reference
      student (distillation-batch-top-log-probs batch)
      (distillation-batch-top-indices batch)
      (distillation-batch-tail-log-prob batch)
      (distillation-batch-ids batch) (distillation-batch-labels batch)
      (distillation-batch-mask batch) (distillation-batch-temperature batch)
      temperature hard-weight function))))

(defun call-with-distillation-loss (student teacher ids labels mask temperature hard-weight function)
  ;; Restrict this first contract to qualified causal graphs, including composed
  ;; stacks. Equal vocabulary size cannot establish equal token-ID meanings.
  (dolist (model (list student teacher)) (validate-distillation-model model))
  (when (eq student teacher)
    (fail 'compatibility-error "Distillation requires a separate frozen teacher"))
  (unless (and (= (config student "vocab_size") (config teacher "vocab_size"))
               (eq (backend-device (model-backend student)) (backend-device (model-backend teacher))))
    (fail 'compatibility-error "Student and teacher require equal vocabulary sizes and the same device"))
  (validate-distillation-options temperature hard-weight)
  (multiple-value-bind (inputs padding targets selected)
      (prepare-training-batch student ids labels mask)
    (declare (ignore targets))
    (let ((*backend* (model-backend student)))
      (with-tensor-scope
        ;; Teacher execution precedes the student autodiff callback. No student
        ;; parameter overrides can leak into a teacher with matching weight names.
        (with-resource (logits (let ((*parameter-overrides* nil))
                                (forward teacher inputs :attention-mask padding)))
          (let* ((flat (reshape logits (list -1 (config teacher "vocab_size"))))
                 (reference (require-finite
                             (take-indices flat (tensor-from-array *backend* selected :dtype :int32)))))
            ;; The returned gradients belong to the caller, not this temporary
            ;; teacher scope. The autodiff callback creates its own tensor scope.
            (call-with-distillation-reference student reference ids labels mask
                                              temperature hard-weight function)))))))

(defun distillation-loss-and-gradients (student teacher ids
                                      &key labels attention-mask (temperature 1.0) (hard-weight 0.0))
  "Return mean temperature-scaled KL(teacher || student) and owned student gradients.
HARD-WEIGHT mixes supervised next-token cross entropy: (1-w)*T^2*KL + w*CE.
Labels -100 and padding select the same next-token positions for both losses.
Models must use the same token-ID meanings; vocabulary size alone cannot prove this.
Only native Llama/composed models on the same device are qualified. Teacher is frozen."
  (call-with-distillation-loss
   student teacher ids labels attention-mask temperature hard-weight
   (lambda () (loss-and-gradients student ids :labels labels :attention-mask attention-mask))))

(defun distill-step (student teacher optimizer ids
                    &key labels attention-mask (temperature 1.0) (hard-weight 0.0) max-grad-norm)
  "Return pre-update distillation loss; atomically update only the student's trainable parameters.
Uses ordinary native optimizer ownership, clipping and checkpoint continuation.
Supply the teacher, data and objective options again after restoring a checkpoint."
  (call-with-distillation-loss
   student teacher ids labels attention-mask temperature hard-weight
   (lambda () (train-step student optimizer ids :labels labels :attention-mask attention-mask
                                               :max-grad-norm max-grad-norm))))

(defun distillation-batch-loss-and-gradients (student batch &key (temperature 1.0) (hard-weight 0.0))
  "Return a reusable batch's loss and owned student gradients without executing a teacher."
  (unless (typep batch 'distillation-batch)
    (fail 'compatibility-error "Expected a DISTILLATION-BATCH"))
  (validate-distillation-model student)
  (pointer (distillation-batch-backend batch))
  (unless (= (distillation-batch-vocab-size batch) (config student "vocab_size"))
    (fail 'compatibility-error "Distillation batch and student vocabulary sizes differ"))
  (call-with-distillation-batch-reference
   student batch temperature hard-weight
   (lambda () (loss-and-gradients student (distillation-batch-ids batch)
                                  :labels (distillation-batch-labels batch)
                                  :attention-mask (distillation-batch-mask batch)))))

(defun distill-batch-step (student batch optimizer
                          &key (temperature 1.0) (hard-weight 0.0) max-grad-norm)
  "Update only STUDENT from a reusable teacher-target batch. Return pre-update loss."
  (unless (typep batch 'distillation-batch)
    (fail 'compatibility-error "Expected a DISTILLATION-BATCH"))
  (validate-distillation-model student)
  (pointer (distillation-batch-backend batch))
  (unless (= (distillation-batch-vocab-size batch) (config student "vocab_size"))
    (fail 'compatibility-error "Distillation batch and student vocabulary sizes differ"))
  (call-with-distillation-batch-reference
   student batch temperature hard-weight
   (lambda () (train-step student optimizer (distillation-batch-ids batch)
                          :labels (distillation-batch-labels batch)
                          :attention-mask (distillation-batch-mask batch)
                          :max-grad-norm max-grad-norm))))

(defun matrix-json-value (array)
  (destructuring-bind (rows columns) (array-dimensions array)
    (coerce (loop for row below rows collect
              (coerce (loop for column below columns collect (aref array row column)) 'vector))
            'vector)))

(defun json-integer-matrix (value field)
  (unless (and (vectorp value) (plusp (length value))
               (every (lambda (row) (and (vectorp row) (plusp (length row))
                                         (= (length row) (length (aref value 0)))
                                         (every (lambda (item) (typep item '(signed-byte 32))) row))) value))
    (fail 'compatibility-error "Distillation ~A must be a nonempty rectangular int32 matrix" field))
  (let ((array (make-array (list (length value) (length (aref value 0))))))
    (loop for row across value for i from 0 do
      (loop for item across row for j from 0 do (setf (aref array i j) item)))
    array))

(defun validate-distillation-artifact-arrays (ids labels mask vocab-size)
  (unless (and (equal (array-dimensions ids) (array-dimensions labels))
               (or (null mask) (equal (array-dimensions ids) (array-dimensions mask)))
               (>= (array-dimension ids 1) 2))
    (fail 'compatibility-error "Distillation IDs, labels, and mask require matching batch shapes with at least two tokens"))
  (dotimes (index (array-total-size ids))
    (unless (typep (row-major-aref ids index) `(integer 0 (,vocab-size)))
      (fail 'compatibility-error "Distillation input ID is outside the teacher vocabulary"))
    (unless (or (eql -100 (row-major-aref labels index))
                (typep (row-major-aref labels index) `(integer 0 (,vocab-size))))
      (fail 'compatibility-error "Distillation label is outside the teacher vocabulary"))
    (when (and mask (not (member (row-major-aref mask index) '(0 1))))
      (fail 'compatibility-error "Distillation attention mask must contain only zero or one")))
  (loop for row below (array-dimension ids 0) sum
    (loop for position from 1 below (array-dimension ids 1)
          count (and (/= -100 (aref labels row position))
                     (or (null mask) (= 1 (aref mask row position)))))))

(defclass distillation-example ()
  ((input-ids :initarg :input-ids :reader distillation-example-input-ids)
   (labels :initarg :labels :reader distillation-example-labels)
   (attention-mask :initarg :attention-mask :reader distillation-example-attention-mask)))

(defun make-distillation-example (input-ids &key labels attention-mask)
  "Snapshot one causal teacher input for memory-bounded dataset production."
  (unless (and (arrayp input-ids) (= (array-rank input-ids) 2)
               (or (null labels) (and (arrayp labels) (= (array-rank labels) 2)))
               (or (null attention-mask)
                   (and (arrayp attention-mask) (= (array-rank attention-mask) 2))))
    (fail 'shape-error "Distillation examples require rank-two input, label, and mask arrays"))
  (let ((ids-copy (copy-array-values input-ids))
        (labels-copy (copy-array-values (or labels input-ids)))
        (mask-copy (and attention-mask (copy-array-values attention-mask))))
    (handler-case
        (unless (plusp (validate-distillation-artifact-arrays
                        ids-copy labels-copy mask-copy 2147483647))
          (fail 'shape-error "Distillation example has no selected next-token positions"))
      (compatibility-error (condition)
        (fail 'shape-error "~A" (error-message condition))))
    (make-instance 'distillation-example :input-ids ids-copy :labels labels-copy
                   :attention-mask mask-copy)))

(defgeneric map-distillation-examples (function source)
  (:documentation "Call FUNCTION once for each DISTILLATION-EXAMPLE in SOURCE, in order.
Applications can specialize this protocol to read a corpus lazily and may discard or reuse
source storage after FUNCTION returns. Consumers complete each example synchronously."))

(defmethod map-distillation-examples (function (source list))
  (dolist (example source source) (funcall function example)))

(defmethod map-distillation-examples (function (source vector))
  (when (stringp source)
    (fail 'shape-error "A string is not a distillation example source"))
  (dotimes (index (length source) source) (funcall function (aref source index))))

(defun validate-stored-distillation-batch (batch &optional expected-vocab-size)
  (unless (typep batch 'distillation-batch)
    (fail 'compatibility-error "Expected a live DISTILLATION-BATCH"))
  (pointer (distillation-batch-backend batch))
  (let* ((vocab-size (distillation-batch-vocab-size batch))
         (selected (validate-distillation-artifact-arrays
                    (distillation-batch-ids batch) (distillation-batch-labels batch)
                    (distillation-batch-mask batch) vocab-size)))
    (unless (and (or (null expected-vocab-size) (= vocab-size expected-vocab-size))
                 (plusp selected))
      (fail 'compatibility-error "Dataset targets require one vocabulary and selected positions"))
    (ecase (distillation-batch-representation batch)
      (:dense
       (let ((logits (distillation-batch-logits batch)))
         (unless (and logits
                      (null (distillation-batch-top-log-probs batch))
                      (null (distillation-batch-top-indices batch))
                      (null (distillation-batch-tail-log-prob batch))
                      (eq (tensor-dtype logits) :float32)
                      (equal (tensor-shape logits) (list selected vocab-size)))
           (fail 'compatibility-error
                 "Dense targets require matching FP32 selected logits"))
         (pointer (require-finite logits))))
      (:top-k
       (let* ((top-log-probs (distillation-batch-top-log-probs batch))
              (indices (distillation-batch-top-indices batch))
              (tail-log-prob (distillation-batch-tail-log-prob batch))
              (top-k (distillation-batch-top-k batch))
              (temperature (distillation-batch-temperature batch)))
         (validate-top-k-distillation-options vocab-size top-k temperature)
         (unless (and (null (distillation-batch-logits batch))
                      top-log-probs indices tail-log-prob
                      (eq (tensor-dtype top-log-probs) :float32)
                      (eq (tensor-dtype indices) :int32)
                      (eq (tensor-dtype tail-log-prob) :float32)
                      (equal (tensor-shape top-log-probs) (list selected top-k))
                      (equal (tensor-shape indices) (list selected top-k))
                      (equal (tensor-shape tail-log-prob) (list selected 1)))
           (fail 'compatibility-error
                 "Top-k targets require matching FP32 probabilities, int32 indices, and tail probabilities"))
         (require-finite top-log-probs)
         (require-finite tail-log-prob)
         (let ((top-values (tensor-array top-log-probs))
               (tail-values (tensor-array tail-log-prob))
               (index-values (tensor-int-array indices)))
           (dotimes (index (array-total-size top-values))
             (when (> (row-major-aref top-values index) 1e-6)
               (fail 'compatibility-error "Top-k teacher log probabilities cannot be positive")))
           (dotimes (index (array-total-size tail-values))
             (when (> (row-major-aref tail-values index) 1e-6)
               (fail 'compatibility-error "Teacher tail log probabilities cannot be positive")))
           (dotimes (row selected)
             (let ((seen (make-hash-table :test 'eql)))
               (dotimes (column top-k)
                 (let ((index (aref index-values row column)))
                   (unless (typep index `(integer 0 (,vocab-size)))
                     (fail 'compatibility-error "Top-k teacher index is outside the vocabulary"))
                   (when (gethash index seen)
                     (fail 'compatibility-error "Top-k teacher indices contain a duplicate"))
                   (setf (gethash index seen) t)))))))))
    selected))

(defun resolve-distillation-storage-dtype (batch requested)
  (let ((dtype (if (and batch (eq requested :preserve))
                   (distillation-batch-storage-dtype batch) requested)))
    (unless (member dtype '(:float32 :float16))
      (fail 'shape-error "Distillation target storage dtype must be :FLOAT32 or :FLOAT16~@[; batch saves also accept :PRESERVE~]"
            batch))
    dtype))

(defun call-with-distillation-storage-targets (batch storage-dtype function)
  (let ((*backend* (distillation-batch-backend batch)))
    (labels ((call-with-float-storage (tensor continuation)
               (ecase storage-dtype
                 (:float32 (funcall continuation tensor))
                 (:float16
                  (with-resource (stored (cast-float16 tensor))
                    (require-finite stored)
                    (checked-status (%tb-tensor-eval (pointer stored)))
                    (funcall continuation stored))))))
      (ecase (distillation-batch-representation batch)
        (:dense
         (call-with-float-storage
          (distillation-batch-logits batch)
          (lambda (logits)
            (funcall function (list (cons "teacher_logits" logits))))))
        (:top-k
         (call-with-float-storage
          (distillation-batch-top-log-probs batch)
          (lambda (top-log-probs)
            (call-with-float-storage
             (distillation-batch-tail-log-prob batch)
             (lambda (tail-log-prob)
               (with-resource
                   (vocabulary
                     (tensor-from-array
                      *backend*
                      (make-array 1 :initial-element
                                  (distillation-batch-vocab-size batch))
                      :dtype :int32))
                 (funcall function
                          (list (cons "teacher_topk_log_probs" top-log-probs)
                                (cons "teacher_topk_indices"
                                      (distillation-batch-top-indices batch))
                                (cons "teacher_tail_log_prob" tail-log-prob)
                                (cons "teacher_vocab_size" vocabulary)))))))))))))

(defun write-distillation-batch-files (batch directory storage-dtype)
  (validate-stored-distillation-batch batch)
  (let* ((representation (distillation-batch-representation batch))
         (weights-path (merge-pathnames "teacher.safetensors" directory))
         (format-version
           (ecase representation
             (:dense (ecase storage-dtype (:float32 1) (:float16 2)))
             (:top-k (ecase storage-dtype (:float32 3) (:float16 4)))))
        (dtype-name (ecase storage-dtype (:float32 "float32") (:float16 "float16"))))
    (ensure-directories-exist weights-path)
    (call-with-distillation-storage-targets
     batch storage-dtype
     (lambda (targets) (save-weights targets weights-path)))
    (let ((manifest (component-object
                     "format" "cl-transformer-blocks-distillation-batch"
                     "format_version" format-version "dtype" dtype-name
                     "vocab_size" (distillation-batch-vocab-size batch)
                     "teacher_source" (distillation-batch-teacher-source batch)
                     "teacher_revision" (distillation-batch-teacher-revision batch)
                     "input_ids" (matrix-json-value (distillation-batch-ids batch))
                     "labels" (matrix-json-value (distillation-batch-labels batch)))))
      (when (eq representation :top-k)
        (setf (gethash "representation" manifest) "top_k"
              (gethash "top_k" manifest) (distillation-batch-top-k batch)
              (gethash "temperature" manifest) (distillation-batch-temperature batch)))
      (when (distillation-batch-mask batch)
        (setf (gethash "attention_mask" manifest)
              (matrix-json-value (distillation-batch-mask batch))))
      (write-json manifest (merge-pathnames "distillation.json" directory)))))

(defun save-distillation-batch (batch destination &key (storage-dtype :preserve))
  "Atomically publish reusable dense or top-k teacher targets in FP32 or FP16.
Floating targets are restored to FP32 on load. By default, preserve source storage dtype."
  (validate-stored-distillation-batch batch)
  (let ((destination (uiop:ensure-directory-pathname destination))
        (storage-dtype (resolve-distillation-storage-dtype batch storage-dtype)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging) (write-distillation-batch-files batch staging storage-dtype)))))

(defun read-distillation-batch-metadata (source)
  (let* ((source (uiop:ensure-directory-pathname source))
         (path (merge-pathnames "distillation.json" source)))
    (unless (and (probe-file path) (probe-file (merge-pathnames "teacher.safetensors" source)))
      (fail 'compatibility-error "Distillation artifact requires distillation.json and teacher.safetensors"))
    (let ((manifest (read-json path)))
      (unless (hash-table-p manifest)
        (fail 'compatibility-error "Distillation metadata must be a JSON object"))
      (let* ((format-version (gethash "format_version" manifest))
             (dtype-name (gethash "dtype" manifest))
             (storage-dtype
               (cond ((and (member format-version '(1 3)) (equal dtype-name "float32")) :float32)
                     ((and (member format-version '(2 4)) (equal dtype-name "float16")) :float16)
                     (t nil)))
             (representation
               (cond ((member format-version '(1 2))
                      (and (null (gethash "representation" manifest))
                           (null (gethash "top_k" manifest))
                           (null (gethash "temperature" manifest))
                           :dense))
                     ((member format-version '(3 4))
                      (and (equal (gethash "representation" manifest) "top_k") :top-k))))
             (vocab-size (gethash "vocab_size" manifest))
             (ids (json-integer-matrix (gethash "input_ids" manifest) "input_ids"))
             (labels (json-integer-matrix (gethash "labels" manifest) "labels"))
             (mask-value (gethash "attention_mask" manifest))
             (mask (and mask-value (json-integer-matrix mask-value "attention_mask")))
             (top-k (and (eq representation :top-k) (gethash "top_k" manifest)))
             (temperature (and (eq representation :top-k) (gethash "temperature" manifest))))
        (unless (and (equal (gethash "format" manifest) "cl-transformer-blocks-distillation-batch")
                     storage-dtype representation
                     (typep vocab-size '(integer 1 2147483647)))
          (fail 'compatibility-error "Unsupported distillation artifact metadata"))
        (when (eq representation :top-k)
          (handler-case
              (validate-top-k-distillation-options vocab-size top-k temperature)
            (shape-error ()
              (fail 'compatibility-error "Unsupported top-k distillation metadata"))))
        (let ((selected (validate-distillation-artifact-arrays ids labels mask vocab-size)))
          (unless (plusp selected)
            (fail 'compatibility-error "Distillation artifact has no selected next-token positions"))
          (values source manifest vocab-size ids labels mask selected storage-dtype
                  representation top-k
                  (and temperature (float temperature 1.0))))))))

(defun load-distillation-batch (source &key (device :cpu))
  "Load a versioned dense or top-k FP32/FP16 target produced by Lisp or Python."
  (multiple-value-bind
      (source manifest vocab-size ids labels mask selected storage-dtype
       representation top-k temperature)
      (read-distillation-batch-metadata source)
        (let ((backend (make-backend :device device))
              (weights (make-hash-table :test 'equal)) (success nil))
          (unwind-protect
               (progn
                 (load-weights (merge-pathnames "teacher.safetensors" source) backend weights
                               :required-dtype (and (eq representation :dense) storage-dtype)
                               :preserve-dtype (eq representation :top-k))
                 (let ((batch
                         (ecase representation
                           (:dense
                            (unless (and (= (hash-table-count weights) 1)
                                         (gethash "teacher_logits" weights)
                                         (equal (tensor-shape (gethash "teacher_logits" weights))
                                                (list selected vocab-size)))
                              (fail 'compatibility-error
                                    "Teacher logits do not match distillation metadata"))
                            (make-instance
                             'distillation-batch :backend backend
                             :logits (gethash "teacher_logits" weights)
                             :ids ids :labels labels :mask mask :vocab-size vocab-size
                             :storage-dtype storage-dtype
                             :teacher-source (gethash "teacher_source" manifest)
                             :teacher-revision (gethash "teacher_revision" manifest)))
                           (:top-k
                            (let ((top (gethash "teacher_topk_log_probs" weights))
                                  (indices (gethash "teacher_topk_indices" weights))
                                  (tail (gethash "teacher_tail_log_prob" weights))
                                  (stored-vocab (gethash "teacher_vocab_size" weights)))
                              (unless (and (= (hash-table-count weights) 4)
                                           top indices tail stored-vocab
                                           (eq (tensor-dtype top) storage-dtype)
                                           (eq (tensor-dtype indices) :int32)
                                           (eq (tensor-dtype tail) storage-dtype)
                                           (eq (tensor-dtype stored-vocab) :int32)
                                           (equal (tensor-shape stored-vocab) '(1))
                                           (= (aref (tensor-int-array stored-vocab) 0)
                                              vocab-size))
                                (fail 'compatibility-error
                                      "Top-k tensor names or storage dtypes do not match metadata"))
                              (unless (eq storage-dtype :float32)
                                (let ((*backend* backend))
                                  (let ((converted (cast-float top)))
                                    (dispose top)
                                    (setf top converted
                                          (gethash "teacher_topk_log_probs" weights) converted))
                                  (let ((converted (cast-float tail)))
                                    (dispose tail)
                                    (setf tail converted
                                          (gethash "teacher_tail_log_prob" weights) converted))))
                              (make-instance
                               'distillation-batch :backend backend :representation :top-k
                               :top-log-probs top :top-indices indices :tail-log-prob tail
                               :top-k top-k :temperature temperature
                               :ids ids :labels labels :mask mask :vocab-size vocab-size
                               :storage-dtype storage-dtype
                               :teacher-source (gethash "teacher_source" manifest)
                               :teacher-revision (gethash "teacher_revision" manifest)))))))
                   (validate-stored-distillation-batch batch)
                   (when (eq representation :top-k)
                     (dispose (gethash "teacher_vocab_size" weights))
                     (remhash "teacher_vocab_size" weights))
                   (setf success t)
                   batch))
            (unless success
              (maphash (lambda (key value) (declare (ignore key)) (dispose value)) weights)
              (dispose backend))))))

(defclass distillation-dataset ()
  ((directory :initarg :directory :reader distillation-dataset-directory)
   (dataset-id :initarg :dataset-id :reader distillation-dataset-id)
   (content-sha256 :initarg :content-sha256 :initform nil
                   :reader distillation-dataset-content-sha256)
   (vocab-size :initarg :vocab-size :reader distillation-dataset-vocab-size)
   (entries :initarg :entries :reader distillation-dataset-entries)
   (device :initarg :device :reader distillation-dataset-device)
   (shuffle :initarg :shuffle :reader distillation-dataset-shuffle-p)
   (seed :initarg :seed :reader distillation-dataset-seed)
   (epoch :initarg :epoch :reader distillation-dataset-epoch)
   (position :initarg :position :reader distillation-dataset-position)
   (order :initarg :order :reader distillation-dataset-order)))

(defun distillation-dataset-size (dataset)
  (unless (typep dataset 'distillation-dataset)
    (fail 'compatibility-error "Expected a DISTILLATION-DATASET"))
  (length (distillation-dataset-entries dataset)))

(defun valid-distillation-dataset-id-p (value)
  (and (stringp value) (<= 1 (length value) 256) (every #'graphic-char-p value)))

(defun distillation-dataset-entry-path (index)
  (format nil "batches/~6,'0D" index))

(defun valid-sha256-string-p (value)
  (and (stringp value) (= (length value) 64)
       (every (lambda (character)
                (or (digit-char-p character)
                    (find character "abcdef" :test #'char=)))
              value)))

(defun make-verified-distillation-dataset-entry (directory index selected)
  (let* ((relative (distillation-dataset-entry-path index))
         (batch-directory (merge-pathnames (format nil "~A/" relative) directory)))
    (component-object
     "path" relative "selected_positions" selected
     "manifest_sha256" (sha256-file (merge-pathnames "distillation.json" batch-directory))
     "weights_sha256" (sha256-file (merge-pathnames "teacher.safetensors" batch-directory)))))

(defun compute-distillation-dataset-content-sha256 (vocab-size entries)
  (sha256-ascii-string
   (with-output-to-string (stream)
     (write-line "cl-transformer-blocks-distillation-dataset-v2" stream)
     (format stream "vocab_size=~D~%batch_count=~D~%" vocab-size (length entries))
     (loop for entry across entries for index from 0 do
       (format stream "~6,'0D~C~D~C~A~C~A~%" index #\Tab
               (gethash "selected_positions" entry) #\Tab
               (gethash "manifest_sha256" entry) #\Tab
               (gethash "weights_sha256" entry))))))

(defun verify-distillation-dataset-entry-content (directory entry index)
  (let* ((batch-directory
           (merge-pathnames (format nil "~A/" (distillation-dataset-entry-path index)) directory))
         (manifest-path (merge-pathnames "distillation.json" batch-directory))
         (weights-path (merge-pathnames "teacher.safetensors" batch-directory)))
    (unless (and (probe-file manifest-path) (probe-file weights-path))
      (fail 'compatibility-error "Distillation dataset batch ~D is incomplete" index))
    (unless (handler-case
                (and (equal (sha256-file manifest-path) (gethash "manifest_sha256" entry))
                     (equal (sha256-file weights-path) (gethash "weights_sha256" entry)))
              (file-error () nil))
      (fail 'compatibility-error
            "Distillation dataset batch ~D does not match its SHA-256 identity" index))))

(defun distillation-dataset-batch-directory (dataset index)
  (merge-pathnames (format nil "~A/" (distillation-dataset-entry-path index))
                   (distillation-dataset-directory dataset)))

(defun make-distillation-dataset-order (count shuffle seed epoch)
  (let ((order (coerce (loop for index below count collect index) 'vector))
        (state (mod (+ seed epoch) (expt 2 32))))
    (when shuffle
      (loop for index downfrom (1- count) above 0 do
        (setf state (mod (+ (* 1664525 state) 1013904223) (expt 2 32)))
        (rotatef (aref order index) (aref order (mod state (1+ index))))))
    order))

(defun validate-distillation-dataset-settings (device shuffle seed)
  (unless (member device '(:cpu :gpu))
    (fail 'compatibility-error "Distillation dataset device must be :CPU or :GPU"))
  (unless (typep shuffle 'boolean)
    (fail 'shape-error "Distillation dataset shuffle must be a boolean"))
  (unless (typep seed '(unsigned-byte 32))
    (fail 'shape-error "Distillation dataset seed must be uint32")))

(defun save-distillation-dataset (batches destination &key dataset-id (storage-dtype :preserve))
  "Atomically publish reusable targets, preserving or selecting each batch's storage dtype."
  (unless (and (typep batches 'sequence) (not (stringp batches)) (plusp (length batches))
               (valid-distillation-dataset-id-p dataset-id))
    (fail 'shape-error "Dataset requires nonempty batches and a printable DATASET-ID"))
  (let* ((batches (coerce batches 'list))
         (vocab-size (and (typep (first batches) 'distillation-batch)
                          (distillation-batch-vocab-size (first batches))))
         (selected-counts
           (mapcar (lambda (batch) (validate-stored-distillation-batch batch vocab-size))
                   batches)))
    (let ((destination (uiop:ensure-directory-pathname destination)))
      (call-with-staged-directory-publication
       destination
       (lambda (staging)
         (let ((entries (make-array (length batches))))
           (loop for batch in batches for selected in selected-counts for index from 0 do
             (write-distillation-batch-files
              batch (merge-pathnames (format nil "~A/" (distillation-dataset-entry-path index))
                                     staging)
              (resolve-distillation-storage-dtype batch storage-dtype))
             (setf (aref entries index)
                   (make-verified-distillation-dataset-entry staging index selected)))
           (write-json
            (component-object
             "format" "cl-transformer-blocks-distillation-dataset"
             "format_version" 2 "dataset_id" dataset-id
             "vocab_size" vocab-size "batch_count" (length batches)
             "content_sha256" (compute-distillation-dataset-content-sha256 vocab-size entries)
             "batches" entries)
            (merge-pathnames "distillation-dataset.json" staging))))))))

(defun save-distillation-dataset-from-teacher
    (teacher source destination
     &key dataset-id (storage-dtype :float32) top-k (temperature 1.0))
  "Atomically stream SOURCE through TEACHER, retaining only one reusable target at a time.
SOURCE follows MAP-DISTILLATION-EXAMPLES and must yield DISTILLATION-EXAMPLE objects.
STORAGE-DTYPE is :FLOAT32 or :FLOAT16; native training restores either to FP32.
TOP-K selects compressed fixed-TEMPERATURE targets with one exact aggregate tail event."
  (validate-distillation-model teacher)
  (unless (valid-distillation-dataset-id-p dataset-id)
    (fail 'shape-error "Streaming dataset production requires a printable DATASET-ID"))
  (setf storage-dtype
        (resolve-distillation-storage-dtype nil storage-dtype))
  (let ((destination (uiop:ensure-directory-pathname destination))
        (vocab-size (config teacher "vocab_size")))
    (if top-k
        (validate-top-k-distillation-options vocab-size top-k temperature)
        (unless (= temperature 1.0)
          (fail 'shape-error "TEMPERATURE applies only when TOP-K is supplied")))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (let ((entries (make-array 0 :adjustable t :fill-pointer 0))
             (index 0))
         (map-distillation-examples
          (lambda (example)
            (unless (typep example 'distillation-example)
              (fail 'compatibility-error
                    "Distillation sources must yield DISTILLATION-EXAMPLE objects"))
            (with-resource
                (batch (if top-k
                           (make-top-k-distillation-batch
                            teacher (distillation-example-input-ids example)
                            :labels (distillation-example-labels example)
                            :attention-mask (distillation-example-attention-mask example)
                            :top-k top-k :temperature temperature)
                           (make-distillation-batch
                            teacher (distillation-example-input-ids example)
                            :labels (distillation-example-labels example)
                            :attention-mask (distillation-example-attention-mask example))))
              (let ((selected (validate-stored-distillation-batch batch vocab-size)))
                (write-distillation-batch-files
                 batch (merge-pathnames
                        (format nil "~A/" (distillation-dataset-entry-path index)) staging)
                 storage-dtype)
                (vector-push-extend
                 (make-verified-distillation-dataset-entry staging index selected)
                 entries)))
            (incf index))
          source)
         (unless (plusp index)
           (fail 'shape-error "Distillation example source produced no batches"))
         (write-json
          (component-object
           "format" "cl-transformer-blocks-distillation-dataset"
           "format_version" 2 "dataset_id" dataset-id
           "vocab_size" vocab-size "batch_count" index
           "content_sha256" (compute-distillation-dataset-content-sha256 vocab-size entries)
           "batches" entries)
          (merge-pathnames "distillation-dataset.json" staging)))))))

(defun load-distillation-dataset (source &key (device :cpu) (shuffle nil) (seed 0))
  "Open a lazy target dataset. Version two verifies exact batch bytes with SHA-256.
Legacy version-one datasets remain readable without a content identity."
  (validate-distillation-dataset-settings device shuffle seed)
  (let* ((source (uiop:ensure-directory-pathname source))
         (path (merge-pathnames "distillation-dataset.json" source)))
    (unless (probe-file path)
      (fail 'compatibility-error "Distillation dataset manifest is missing"))
    (let* ((manifest (read-json path))
           (dataset-id (and (hash-table-p manifest) (gethash "dataset_id" manifest)))
           (format-version (and (hash-table-p manifest) (gethash "format_version" manifest)))
           (vocab-size (and (hash-table-p manifest) (gethash "vocab_size" manifest)))
           (batch-count (and (hash-table-p manifest) (gethash "batch_count" manifest)))
           (entries (and (hash-table-p manifest) (gethash "batches" manifest)))
           (content-sha256 (and (eql format-version 2) (gethash "content_sha256" manifest))))
      (unless (and (hash-table-p manifest)
                   (equal (gethash "format" manifest)
                          "cl-transformer-blocks-distillation-dataset")
                   (member format-version '(1 2))
                   (valid-distillation-dataset-id-p dataset-id)
                   (typep vocab-size '(integer 1 2147483647))
                   (typep batch-count '(integer 1))
                   (vectorp entries) (= (length entries) batch-count)
                   (or (eql format-version 1) (valid-sha256-string-p content-sha256)))
        (fail 'compatibility-error "Unsupported distillation dataset metadata"))
      (loop for entry across entries for index from 0 do
        (unless (and (hash-table-p entry)
                     (equal (gethash "path" entry) (distillation-dataset-entry-path index))
                     (typep (gethash "selected_positions" entry) '(integer 1))
                     (or (eql format-version 1)
                         (and (valid-sha256-string-p (gethash "manifest_sha256" entry))
                              (valid-sha256-string-p (gethash "weights_sha256" entry)))))
          (fail 'compatibility-error "Distillation dataset entry metadata is invalid"))
        (when (eql format-version 2)
          (verify-distillation-dataset-entry-content source entry index))
        (multiple-value-bind
            (batch-source batch-manifest batch-vocab ids labels mask selected storage-dtype)
            (read-distillation-batch-metadata
             (merge-pathnames (format nil "~A/" (distillation-dataset-entry-path index)) source))
          (declare (ignore batch-source batch-manifest ids labels mask storage-dtype))
          (unless (and (= batch-vocab vocab-size)
                       (= selected (gethash "selected_positions" entry)))
            (fail 'compatibility-error "Distillation dataset entry disagrees with its batch"))))
      (when (and (eql format-version 2)
                 (not (equal content-sha256
                             (compute-distillation-dataset-content-sha256 vocab-size entries))))
        (fail 'compatibility-error "Distillation dataset content identity is invalid"))
      (make-instance 'distillation-dataset :directory source :dataset-id dataset-id
                     :content-sha256 content-sha256 :vocab-size vocab-size
                     :entries entries :device device
                     :shuffle shuffle :seed seed :epoch 0 :position 0
                     :order (make-distillation-dataset-order batch-count shuffle seed 0)))))

(defun start-distillation-dataset-epoch (dataset epoch)
  "Set an explicit uint32 epoch and rewind to its deterministic first batch."
  (distillation-dataset-size dataset)
  (unless (typep epoch '(unsigned-byte 32))
    (fail 'shape-error "Distillation dataset epoch must be uint32"))
  (setf (slot-value dataset 'epoch) epoch
        (slot-value dataset 'position) 0
        (slot-value dataset 'order)
        (make-distillation-dataset-order (distillation-dataset-size dataset)
                                         (distillation-dataset-shuffle-p dataset)
                                         (distillation-dataset-seed dataset) epoch))
  dataset)

(defun load-distillation-dataset-batch (dataset index)
  "Load one physical dataset index as an owned reusable target.
Verified datasets recheck the exact batch files immediately before every lazy load."
  (let ((size (distillation-dataset-size dataset)))
    (unless (typep index `(integer 0 (,size)))
      (fail 'shape-error "Distillation dataset index is out of range"))
    (when (distillation-dataset-content-sha256 dataset)
      (verify-distillation-dataset-entry-content
       (distillation-dataset-directory dataset)
       (aref (distillation-dataset-entries dataset) index) index))
    (let ((batch (load-distillation-batch (distillation-dataset-batch-directory dataset index)
                                          :device (distillation-dataset-device dataset))))
      (unless (= (distillation-batch-vocab-size batch)
                 (distillation-dataset-vocab-size dataset))
        (dispose batch)
        (fail 'compatibility-error "Loaded target vocabulary differs from its dataset"))
      batch)))

(defun current-distillation-dataset-index (dataset)
  (let ((position (distillation-dataset-position dataset)))
    (and (< position (distillation-dataset-size dataset))
         (aref (distillation-dataset-order dataset) position))))

(defun next-distillation-dataset-batch (dataset)
  "Load and consume the next target. Return its physical index as a second value, or NIL at end."
  (let ((index (current-distillation-dataset-index dataset)))
    (when index
      (let ((batch (load-distillation-dataset-batch dataset index)))
        (incf (slot-value dataset 'position))
        (values batch index)))))

(defun distill-dataset-step (student dataset optimizer
                             &key (temperature 1.0) (hard-weight 0.0) max-grad-norm)
  "Update STUDENT from the next target, consuming it only after a successful step.
Return the loss and physical batch index, or NIL without changing state at epoch end."
  (let ((index (current-distillation-dataset-index dataset)))
    (when index
      (with-resource (batch (load-distillation-dataset-batch dataset index))
        (let ((loss (distill-batch-step student batch optimizer :temperature temperature
                                        :hard-weight hard-weight :max-grad-norm max-grad-norm)))
          (incf (slot-value dataset 'position))
          (values loss index))))))

(defun save-distillation-dataset-state (dataset destination)
  "Atomically save the next unconsumed position for exact iterator continuation.
Version-two state binds the cursor to the dataset's verified content identity."
  (distillation-dataset-size dataset)
  (let ((destination (uiop:ensure-directory-pathname destination)))
    (call-with-staged-directory-publication
     destination
     (lambda (staging)
       (let ((manifest
               (component-object
                "format" "cl-transformer-blocks-distillation-dataset-state"
                "format_version" (if (distillation-dataset-content-sha256 dataset) 2 1)
                "dataset_id" (distillation-dataset-id dataset)
                "vocab_size" (distillation-dataset-vocab-size dataset)
                "batch_count" (distillation-dataset-size dataset)
                "shuffle" (if (distillation-dataset-shuffle-p dataset)
                              'yason:true 'yason:false)
                "seed" (distillation-dataset-seed dataset)
                "epoch" (distillation-dataset-epoch dataset)
                "position" (distillation-dataset-position dataset))))
         (when (distillation-dataset-content-sha256 dataset)
           (setf (gethash "content_sha256" manifest)
                 (distillation-dataset-content-sha256 dataset)))
         (write-json manifest (merge-pathnames "dataset-state.json" staging)))))))

(defun restore-distillation-dataset-state (dataset source)
  "Restore a saved iterator only when dataset content and iteration settings match."
  (distillation-dataset-size dataset)
  (let* ((source (uiop:ensure-directory-pathname source))
         (path (merge-pathnames "dataset-state.json" source)))
    (unless (probe-file path)
      (fail 'compatibility-error "Distillation dataset state manifest is missing"))
    (let* ((manifest (read-json path))
           (shuffle-value (and (hash-table-p manifest) (gethash "shuffle" manifest)))
           (epoch (and (hash-table-p manifest) (gethash "epoch" manifest)))
           (position (and (hash-table-p manifest) (gethash "position" manifest)))
           (content-sha256 (distillation-dataset-content-sha256 dataset))
           (expected-version (if content-sha256 2 1)))
      (unless (and (hash-table-p manifest)
                   (equal (gethash "format" manifest)
                          "cl-transformer-blocks-distillation-dataset-state")
                   (eql (gethash "format_version" manifest) expected-version)
                   (equal (gethash "dataset_id" manifest) (distillation-dataset-id dataset))
                   (or (null content-sha256)
                       (equal (gethash "content_sha256" manifest) content-sha256))
                   (= (gethash "vocab_size" manifest 0)
                      (distillation-dataset-vocab-size dataset))
                   (= (gethash "batch_count" manifest 0) (distillation-dataset-size dataset))
                   (member shuffle-value '(yason:true yason:false))
                   (eql (json-true-p shuffle-value) (distillation-dataset-shuffle-p dataset))
                   (eql (gethash "seed" manifest) (distillation-dataset-seed dataset))
                   (typep epoch '(unsigned-byte 32))
                   (typep position `(integer 0 ,(distillation-dataset-size dataset))))
        (fail 'compatibility-error "Dataset state does not match this dataset or iterator"))
      (start-distillation-dataset-epoch dataset epoch)
      (setf (slot-value dataset 'position) position)
      position)))
