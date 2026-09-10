(in-package #:tb)
(defvar *training-model* nil)
(defvar *training-inputs* nil)
(defvar *training-mask* nil)
(defvar *training-token-type-ids* nil)
(defvar *training-targets* nil)
(defvar *training-selected* nil)
(defvar *training-names* nil)
(defvar *training-error* nil)
(defvar *training-loss-function* #'cross-entropy)
(define-native "tb_value_and_grad" :int (callback :pointer) (payload :pointer)
               (inputs :pointer) (count :int) (loss :pointer) (grads :pointer))
(cffi:defcallback training-loss :int ((out :pointer) (inputs :pointer) (count :int) (payload :pointer))
  (declare (ignore payload))
  (handler-case
      (let ((*parameter-overrides* (make-hash-table :test 'equal)))
        (with-tensor-scope
          (loop for name in *training-names* for i below count do
            (setf (gethash name *parameter-overrides*)
                  (wrap-tensor (%tb-tensor-retain (cffi:mem-aref inputs :pointer i)) *backend*)))
          (let* ((logits (compute-logits *training-model*
                          (tensor-from-array *backend* *training-inputs* :dtype :int32)
                          *training-mask* 0 nil
                          (when *training-token-type-ids*
                            (tensor-from-array *backend* *training-token-type-ids* :dtype :int32))))
                 (flat (reshape logits (list -1 (config *training-model* "vocab_size"))))
                 (selected (take-indices flat (tensor-from-array *backend* *training-selected* :dtype :int32)))
                 (loss (funcall *training-loss-function* selected
                                (tensor-from-array *backend* *training-targets* :dtype :int32))))
            (setf (cffi:mem-ref out :pointer) (checked-pointer (%tb-tensor-retain (pointer loss)))))) 0)
    (error (condition) (setf *training-error* condition) 1)))
(defmethod prepare-training-batch ((model model) ids labels mask)
  (pointer (model-backend model)) (validate-training-config model) (validate-inputs model ids mask nil)
  (let* ((b (array-dimension ids 0)) (n (array-dimension ids 1))
         (labels (or labels ids)) (targets nil) (selected nil))
    (when (< n 2) (fail 'shape-error "Training requires at least two tokens"))
    (unless (and (arrayp labels) (equal (array-dimensions labels) (list b n)))
      (fail 'shape-error "Labels must have the input shape; -100 ignores a target"))
    (dotimes (i (array-total-size labels))
      (unless (or (eql -100 (row-major-aref labels i))
                  (typep (row-major-aref labels i) `(integer 0 (,(config model "vocab_size")))))
        (fail 'shape-error "Label out of vocabulary")))
    (let ((x (make-array (list b (1- n)))) (padding (when mask (make-array (list b (1- n))))))
      (dotimes (batch b) (dotimes (i (1- n))
        (setf (aref x batch i) (aref ids batch i))
        (when mask (setf (aref padding batch i) (aref mask batch i)))
        (let ((target (aref labels batch (1+ i))))
          (when (and (/= target -100) (or (null mask) (= 1 (aref mask batch (1+ i)))))
            (push target targets) (push (+ (* batch (1- n)) i) selected)))))
      (unless targets (fail 'shape-error "No supervised next-token targets"))
      (values x padding (coerce (nreverse targets) 'vector) (coerce (nreverse selected) 'vector)))))
(defun loss-and-gradients (model ids &key labels attention-mask token-type-ids)
  "Mean task loss and owned gradients for TRAINABLE-PARAMETERS.
Labels use -100 to ignore targets. Decoder models shift labels and honor padding in
loss selection; masked-language models select explicitly labeled input positions.
TOKEN-TYPE-IDS supplies BERT segment IDs with the same shape as IDS."
  (unless (typep model 'model)
    (fail 'compatibility-error
          "Direct Python gradient transfer is unavailable; use PYTHON-TRAIN-STEP"))
  (multiple-value-bind (*training-inputs* *training-mask* *training-targets* *training-selected*)
      (prepare-training-batch model ids labels attention-mask)
    (validate-token-type-ids model ids token-type-ids)
    (let* ((*backend* (model-backend model)) (*training-model* model)
           (*training-token-type-ids* token-type-ids)
           (params (trainable-parameters model)) (*training-names* (mapcar #'car params))
           (*training-error* nil) (count (length params)) (result nil) (success nil))
      (cffi:with-foreign-objects ((inputs :pointer count) (grads :pointer count) (loss :pointer))
        (loop for entry in params for i from 0 do (setf (cffi:mem-aref inputs :pointer i) (pointer (cdr entry))))
        (let ((status (%tb-value-and-grad (cffi:callback training-loss) (cffi:null-pointer) inputs count loss grads)))
          (when *training-error* (error *training-error*)) (checked-status status))
        (with-resource (value (wrap-tensor (cffi:mem-ref loss :pointer) *backend*))
          (unwind-protect
               (progn
                 (loop for name in *training-names* for i from 0
                       do (push (cons name (wrap-tensor (cffi:mem-aref grads :pointer i) *backend*)) result))
                 (require-finite value)
                 (dolist (entry result) (require-finite (cdr entry)))
                 (setf success t)
                 (values (row-major-aref (tensor-array value) 0) (nreverse result)))
            (unless success (mapc (lambda (entry) (dispose (cdr entry))) result))))))))
(defun sgd-step (model ids &key (learning-rate 0.01) labels attention-mask token-type-ids)
  "One stateless SGD step over the active trainable parameters. Return pre-update loss."
  (with-resource (optimizer (make-sgd :learning-rate learning-rate))
    (train-step model optimizer ids :labels labels :attention-mask attention-mask
                                   :token-type-ids token-type-ids)))
