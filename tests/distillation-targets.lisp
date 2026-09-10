(in-package #:tb-tests)

(defun target-gradient-arrays (gradients)
  (mapcar (lambda (entry) (cons (car entry) (tb:tensor-array (cdr entry)))) gradients))

(defun check-target-gradients (actual expected tolerance)
  (check (= (length actual) (length expected)) "cached target differentiates every student parameter")
  (loop for entry in actual for reference in expected do
    (check (equal (car entry) (car reference)) "cached target gradient names agree")
    (close-arrays (tb:tensor-array (cdr entry)) (cdr reference) tolerance 3e-4)))

(defun check-distillation-target-rejections (batch student output)
  (signals tb:compatibility-error (tb:distillation-batch-loss-and-gradients student nil))
  (signals tb:shape-error (tb:distillation-batch-loss-and-gradients student batch :temperature 0))
  (tb:with-resource (optimizer (tb:make-sgd))
    (signals tb:shape-error (tb:distill-batch-step student batch optimizer :hard-weight 2)))
  (let ((sentinel (merge-pathnames "sentinel" output)))
    (ensure-directories-exist sentinel)
    (with-open-file (stream sentinel :direction :output :if-exists :supersede) (write-string "old" stream))
    (let ((tb::*before-directory-publication*
            (lambda (stage destination) (declare (ignore stage destination)) (error "injected target failure"))))
      (signals error (tb:save-distillation-batch batch output)))
    (check (equal (uiop:read-file-string sentinel) "old") "failed target publication preserves old artifact")))

(defun activate-target-student (student real student-path)
  (when real (tb:load-adapter student (merge-pathnames "adapter/" student-path)))
  student)

(defun check-distillation-storage-overflow (directory device)
  (let ((baseline (tb::%tb-live-tensors))
        (backend (tb:make-backend :device device))
        (batch nil))
    (unwind-protect
         (let ((tb::*backend* backend)
               (sentinel (merge-pathnames "sentinel" directory)))
           (setf batch
                 (make-instance
                  'tb:distillation-batch :backend backend
                  :logits (tb:tensor-from-array backend #2A((70000.0)))
                  :ids #2A((0 0)) :labels #2A((-100 0)) :mask nil :vocab-size 1
                  :teacher-source "overflow-fixture" :teacher-revision nil))
           (ensure-directories-exist sentinel)
           (with-open-file (stream sentinel :direction :output :if-exists :supersede)
             (write-string "old-overflow" stream))
           (signals tb:backend-error
             (tb:save-distillation-batch batch directory :storage-dtype :float16))
           (check (equal (uiop:read-file-string sentinel) "old-overflow")
                  "FP16 overflow preserves the previous artifact"))
      (if batch (tb:dispose batch) (tb:dispose backend)))
    (check (= baseline (tb::%tb-live-tensors))
           "FP16 overflow releases its temporary cast tensor")))

(defun check-distillation-artifact-rejections (directory device)
  (let* ((path (merge-pathnames "distillation.json" directory))
         (original (uiop:read-file-string path)))
    (unwind-protect
         (dolist (mutation
                   (list (lambda (manifest) (setf (gethash "format_version" manifest) "1"))
                         (lambda (manifest)
                           (setf (gethash "format_version" manifest)
                                 (ecase (gethash "format_version" manifest)
                                   (1 2) (2 1) (3 4) (4 3))))
                         (lambda (manifest)
                           (setf (gethash "dtype" manifest)
                                 (if (equal (gethash "dtype" manifest) "float32")
                                     "float16" "float32")))
                         (lambda (manifest) (setf (gethash "labels" manifest)
                                                  #(#(-100 -100 -100 -100 -100)
                                                    #(-100 -100 -100 -100 -100))))
                         (lambda (manifest) (setf (aref (aref (gethash "input_ids" manifest) 0) 0) -1))
                         (lambda (manifest) (setf (aref (aref (gethash "attention_mask" manifest) 0) 0) 2))
                         (lambda (manifest) (incf (gethash "vocab_size" manifest)))
                         (lambda (manifest)
                           (if (gethash "top_k" manifest)
                               (setf (gethash "top_k" manifest) 0)
                               (setf (gethash "top_k" manifest) 1)))
                         (lambda (manifest)
                           (if (gethash "temperature" manifest)
                               (setf (gethash "temperature" manifest) 0)
                               (setf (gethash "temperature" manifest) 2.0)))))
           (let ((manifest (tb::read-json path)))
             (funcall mutation manifest)
             (tb::write-json manifest path)
             (signals tb:compatibility-error (tb:load-distillation-batch directory :device device))
             (with-open-file (stream path :direction :output :if-exists :supersede
                                          :external-format :utf-8)
               (write-string original stream))))
      (with-open-file (stream path :direction :output :if-exists :supersede :external-format :utf-8)
        (write-string original stream)))))

(defun run-distillation-target-tests ()
  (check (fboundp 'tb::make-top-k-distillation-batch)
         "top-k distillation target constructor is available")
  (tb::ensure-native)
  (let* ((device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (real (equal (uiop:getenv "TB_DISTILLATION_TARGETS_REAL") "1"))
         (root (asdf:system-source-directory "cl-transformer-blocks"))
         (teacher-path (if real (merge-pathnames ".build/models/smollm2/" root)
                           (merge-pathnames (format nil ".build/components-created-~(~A~)/" device) root)))
         (student-path (merge-pathnames (if real (format nil ".build/recomposition-~(~A~)/real/" device)
                                           (format nil ".build/recomposition-~(~A~)/untied/drop/" device)) root))
         (target-root (merge-pathnames (format nil ".build/distillation-targets-~(~A~)/~A"
                                               device (if real "real/" "")) root))
         (output (merge-pathnames "native/" target-root))
         (half-output (merge-pathnames "native-fp16/" target-root))
         (half-resaved-output (merge-pathnames "native-fp16-resaved/" target-root))
         (top-output (merge-pathnames "native-topk/" target-root))
         (top-half-output (merge-pathnames "native-topk-fp16/" target-root))
         (top-half-resaved-output (merge-pathnames "native-topk-fp16-resaved/" target-root))
         (python-output (merge-pathnames "python/" target-root))
         (python-half-output (merge-pathnames "python-fp16/" target-root))
         (python-top-output (merge-pathnames "python-topk/" target-root))
         (python-top-half-output (merge-pathnames "python-topk-fp16/" target-root))
         (foreign-output (merge-pathnames "python-foreign/" target-root))
         (ids #2A((3 4 5 6 0) (7 5 4 3 8)))
         (mask #2A((1 1 1 1 0) (1 1 1 1 1)))
         (labels #2A((-100 -100 5 6 -100) (-100 5 -100 3 8)))
         (top-k (if real 64 4))
         (reference-loss nil) (reference-gradients nil) (batch nil)
         (top-reference-loss nil) (top-reference-gradients nil) (top-batch nil)
         (tolerance (if real 1e-3 2e-6))
         (half-tolerance (if real 1e-2 1e-3))
         (handles (tb::%tb-live-tensors)))
    (tb:with-resource (teacher (tb:from-pretrained teacher-path :device device))
      (tb:with-resource (student (tb:from-pretrained student-path :device device))
        (activate-target-student student real student-path)
        (check (member :distillation-targets (getf (tb:model-capabilities student) :operations))
               "qualified student advertises reusable distillation targets")
        (multiple-value-bind (loss gradients)
            (tb:distillation-loss-and-gradients student teacher ids :labels labels
                                                :attention-mask mask :temperature 2.0 :hard-weight 0.3)
          (unwind-protect
               (setf reference-loss loss reference-gradients (target-gradient-arrays gradients))
            (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients))))
      (setf batch (tb:make-distillation-batch teacher ids :labels labels :attention-mask mask)
            top-batch (tb:make-top-k-distillation-batch
                       teacher ids :labels labels :attention-mask mask
                       :top-k top-k :temperature 2.0))
      (signals tb:shape-error
        (tb:make-top-k-distillation-batch teacher ids :top-k 0))
      (signals tb:shape-error
        (tb:make-top-k-distillation-batch
         teacher ids :top-k (gethash "vocab_size" (tb:model-config teacher)))))
    (check (eq (tb:distillation-batch-storage-dtype batch) :float32)
           "fresh native target records FP32 storage provenance")
    (check (and (eq (tb:distillation-batch-representation top-batch) :top-k)
                (= (tb:distillation-batch-top-k top-batch) top-k)
                (= (tb:distillation-batch-temperature top-batch) 2.0))
           "fresh sparse target records its top-k contract")
    (setf (aref ids 0 0) 9 (aref labels 0 2) -100 (aref mask 1 4) 0)
    (unwind-protect
         (progn
           (tb:with-resource (student (tb:from-pretrained student-path :device device))
             (activate-target-student student real student-path)
             (multiple-value-bind (loss gradients)
                 (tb:distillation-batch-loss-and-gradients student batch :temperature 2.0 :hard-weight 0.3)
               (unwind-protect
                    (progn (check (= loss reference-loss) "in-memory target loss equals live teacher exactly")
                           (check-target-gradients gradients reference-gradients 0))
                 (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients)))
             (check-distillation-target-rejections batch student output))
           (tb:with-resource (student (tb:from-pretrained student-path :device device))
             (activate-target-student student real student-path)
             (multiple-value-bind (loss gradients)
                 (tb:distillation-batch-loss-and-gradients
                  student top-batch :temperature 2.0 :hard-weight 0.3)
               (unwind-protect
                    (setf top-reference-loss loss
                          top-reference-gradients (target-gradient-arrays gradients))
                 (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients)))
             (signals tb:compatibility-error
               (tb:distillation-batch-loss-and-gradients
                student top-batch :temperature 1.0 :hard-weight 0.3)))
           (tb:save-distillation-batch batch output)
           (tb:save-distillation-batch batch half-output :storage-dtype :float16)
           (tb:save-distillation-batch top-batch top-output)
           (tb:save-distillation-batch top-batch top-half-output :storage-dtype :float16)
           (signals tb:shape-error
             (tb:save-distillation-batch batch half-output :storage-dtype :bfloat16)))
      (tb:dispose batch)
      (tb:dispose top-batch))
    (unless real
      (tb:with-resource (unsupported
                          (tb:from-pretrained
                           (merge-pathnames (format nil ".build/portable-lisp-created-~(~A~)/" device) root)
                           :device device))
        (tb:with-resource (student (tb:from-pretrained student-path :device device))
          (signals tb:compatibility-error
            (tb:distillation-loss-and-gradients student unsupported ids)))))
    (dolist (case
              (remove-if-not
               (lambda (entry) (uiop:directory-exists-p (getf entry :path)))
               (list (list :path output :reference :dense :storage :float32 :tolerance tolerance)
                     (list :path half-output :reference :dense :storage :float16
                           :tolerance half-tolerance :interop t
                           :resave half-resaved-output)
                     (list :path top-output :reference :top-k :representation :top-k
                           :storage :float32 :tolerance tolerance)
                     (list :path top-half-output :reference :top-k :representation :top-k
                           :storage :float16 :tolerance half-tolerance :interop t
                           :resave top-half-resaved-output)
                     (list :path python-output :reference :dense :storage :float32
                           :tolerance tolerance :interop t)
                     (list :path python-half-output :reference :dense :storage :float16
                           :tolerance half-tolerance :interop t)
                     (list :path python-top-output :reference :top-k :representation :top-k
                           :storage :float32 :tolerance tolerance :interop t)
                     (list :path python-top-half-output :reference :top-k :representation :top-k
                           :storage :float16 :tolerance half-tolerance :interop t)
                     (list :path foreign-output :reference nil :storage :float32
                           :tolerance tolerance :interop t))))
      (let ((artifact (getf case :path)) (reference-kind (getf case :reference))
            (artifact-tolerance (getf case :tolerance)))
      (tb:with-resource (loaded (tb:load-distillation-batch artifact :device device))
        (check (eq (tb:distillation-batch-storage-dtype loaded) (getf case :storage))
               "loaded target records its artifact storage dtype")
        (check (eq (tb:distillation-batch-representation loaded)
                   (or (getf case :representation) :dense))
               "loaded target records its representation")
        (if (eq (tb:distillation-batch-representation loaded) :dense)
            (check (eq (tb:tensor-dtype (tb::distillation-batch-logits loaded)) :float32)
                   "loaded dense target exposes FP32 logits")
            (progn
              (check (and (= (tb:distillation-batch-top-k loaded) top-k)
                          (= (tb:distillation-batch-temperature loaded) 2.0))
                     "loaded sparse target preserves k and temperature")
              (check (and (eq (tb:tensor-dtype (tb::distillation-batch-top-log-probs loaded))
                              :float32)
                          (eq (tb:tensor-dtype (tb::distillation-batch-top-indices loaded))
                              :int32)
                          (eq (tb:tensor-dtype (tb::distillation-batch-tail-log-prob loaded))
                              :float32))
                     "loaded sparse target exposes FP32 probabilities and int32 indices")))
        (when (getf case :resave)
          (tb:save-distillation-batch loaded (getf case :resave)))
        (tb:with-resource (student (tb:from-pretrained student-path :device device))
          (activate-target-student student real student-path)
          (multiple-value-bind (loss gradients)
              (tb:distillation-batch-loss-and-gradients student loaded :temperature 2.0 :hard-weight 0.3)
            (unwind-protect
                 (progn (check (realp loss) "portable target returns a finite scalar loss")
                        (when reference-kind
                          (let ((expected-loss (if (eq reference-kind :dense)
                                                   reference-loss top-reference-loss))
                                (expected-gradients (if (eq reference-kind :dense)
                                                        reference-gradients
                                                        top-reference-gradients)))
                          (check (< (abs (- loss expected-loss)) artifact-tolerance)
                                 "portable target loss matches live teacher")
                          (check-target-gradients gradients expected-gradients artifact-tolerance)))
                        (when (getf case :interop)
                          (tb::save-weights gradients (merge-pathnames "native-gradients.safetensors" artifact))))
              (mapc (lambda (entry) (tb:dispose (cdr entry))) gradients)))
          (tb:with-resource (optimizer (tb:make-sgd :learning-rate 0.01 :momentum 0.9 :weight-decay 0.02))
            (dotimes (step 3)
              (declare (ignore step))
              (tb:distill-batch-step student loaded optimizer :temperature 2.0 :hard-weight 0.3 :max-grad-norm 0.05))
            (when (getf case :interop)
              (if real (tb:save-adapter student (merge-pathnames "native-updated/" artifact))
                  (tb:save-pretrained student (merge-pathnames "native-updated/" artifact)))))))))
    (check-distillation-artifact-rejections output device)
    (check-distillation-artifact-rejections half-output device)
    (check-distillation-artifact-rejections top-output device)
    (check-distillation-artifact-rejections top-half-output device)
    (signals tb:compatibility-error
      (tb:load-distillation-batch (merge-pathnames "malformed-fp16/" target-root) :device device))
    (signals tb:compatibility-error
      (tb:load-distillation-batch (merge-pathnames "malformed-fp32-v2/" target-root)
                                  :device device))
    (signals tb:compatibility-error
      (tb:load-distillation-batch (merge-pathnames "malformed-topk-duplicate/" target-root)
                                  :device device))
    (signals tb:compatibility-error
      (tb:load-distillation-batch (merge-pathnames "malformed-topk-range/" target-root)
                                  :device device))
    (check-distillation-storage-overflow
     (merge-pathnames "overflow-publication/" target-root) device)
    (check (= handles (tb::%tb-live-tensors)) "all reusable-target resources return to baseline")
    (format t "~&Reusable distillation targets passed: ~A, real=~S (~D checks).~%" device real *checks*)))
