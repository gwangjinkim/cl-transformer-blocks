(in-package #:tb-tests)

(defparameter *dataset-cases*
  (list (list #2A((3 4 5 6 0) (7 5 4 3 8))
              #2A((-100 -100 5 6 -100) (-100 5 -100 3 8))
              #2A((1 1 1 1 0) (1 1 1 1 1)))
        (list #2A((4 3 7 5)) #2A((-100 3 -100 5)) #2A((1 1 1 1)))
        (list #2A((8 6 4 0 0)) #2A((-100 -100 4 -100 -100)) #2A((1 1 1 0 0)))))

(defun ascii-octets (string)
  (let ((octets (make-array (length string) :element-type '(unsigned-byte 8))))
    (dotimes (index (length string) octets)
      (setf (aref octets index) (char-code (char string index))))))

(defun read-file-octets (path)
  (with-open-file (stream path :direction :input :element-type '(unsigned-byte 8))
    (let ((octets (make-array (file-length stream) :element-type '(unsigned-byte 8))))
      (read-sequence octets stream)
      octets)))

(defun write-file-octets (path octets)
  (with-open-file (stream path :direction :output :if-exists :supersede
                               :element-type '(unsigned-byte 8))
    (write-sequence octets stream)))

(defun check-sha256-vectors ()
  (check (equal (tb::sha256-octets (ascii-octets ""))
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
         "SHA-256 empty vector")
  (check (equal (tb::sha256-octets (ascii-octets "abc"))
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
         "SHA-256 abc vector")
  (check (equal (tb::sha256-octets
                 (ascii-octets
                  "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"))
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1")
         "SHA-256 multi-block vector"))

(defclass test-distillation-source ()
  ((examples :initarg :examples :reader test-distillation-source-examples)
   (after-yield :initarg :after-yield :reader test-distillation-source-after-yield)))

(defmethod tb:map-distillation-examples (function (source test-distillation-source))
  (dolist (example (test-distillation-source-examples source) source)
    (funcall function example)
    (funcall (test-distillation-source-after-yield source))))

(defun overwrite-array (array value)
  (dotimes (index (array-total-size array) array)
    (setf (row-major-aref array index) value)))

(defun make-streaming-examples ()
  (mapcar
   (lambda (case)
     (destructuring-bind (original-ids original-labels original-mask) case
       (let* ((ids (tb::copy-array-values original-ids))
              (labels (tb::copy-array-values original-labels))
              (mask (tb::copy-array-values original-mask))
              (example (tb:make-distillation-example
                        ids :labels labels :attention-mask mask)))
         ;; The producer must use the constructor's snapshots, not caller arrays.
         (overwrite-array ids 31)
         (overwrite-array labels -100)
         (overwrite-array mask 0)
         example)))
   *dataset-cases*))

(defun dispose-targets (targets)
  (mapc #'tb:dispose targets))

(defun activate-dataset-student (student real student-path)
  (when real (tb:load-adapter student (merge-pathnames "adapter/" student-path)))
  student)

(defun make-dataset-targets (teacher &key top-k (temperature 1.0))
  (mapcar (lambda (case)
            (destructuring-bind (ids labels mask) case
              (if top-k
                  (tb:make-top-k-distillation-batch
                   teacher ids :labels labels :attention-mask mask
                   :top-k top-k :temperature temperature)
                  (tb:make-distillation-batch
                   teacher ids :labels labels :attention-mask mask))))
          *dataset-cases*))

(defun drain-dataset (student dataset optimizer)
  (loop for loss = (tb:distill-dataset-step
                    student dataset optimizer :temperature 2.0 :hard-weight 0.3
                    :max-grad-norm 0.05)
        while loss collect loss))

(defun parameter-arrays (model)
  (mapcar (lambda (entry) (cons (car entry) (tb:tensor-array (cdr entry))))
          (tb:trainable-parameters model)))

(defun check-parameter-arrays (model expected)
  (loop for entry in (tb:trainable-parameters model) for reference in expected do
    (check (equal (car entry) (car reference)) "resumed dataset parameter names agree")
    (close-arrays (tb:tensor-array (cdr entry)) (cdr reference) 0 0)))

(defun check-python-dataset
    (native-directory python-directory student-path device real tolerance storage-dtype)
  (when (equal (uiop:getenv "TB_DISTILLATION_DATASET_PYTHON") "1")
    (let ((native (tb:load-distillation-dataset native-directory :device device))
          (python (tb:load-distillation-dataset python-directory :device device)))
      (check (= (tb:distillation-dataset-size native) (tb:distillation-dataset-size python))
             "Python/native dataset sizes agree")
      (dotimes (index (tb:distillation-dataset-size native))
        (tb:with-resource (native-batch (tb:load-distillation-dataset-batch native index))
          (tb:with-resource (python-batch (tb:load-distillation-dataset-batch python index))
            (check (eq (tb:distillation-batch-storage-dtype native-batch) storage-dtype)
                   "native dataset batch records its storage dtype")
            (check (eq (tb:distillation-batch-storage-dtype python-batch) storage-dtype)
                   "Python dataset batch records its storage dtype")
            (check (eq (tb:distillation-batch-representation native-batch)
                       (tb:distillation-batch-representation python-batch))
                   "Python/native dataset representations agree")
            (tb:with-resource (student (tb:from-pretrained student-path :device device))
              (activate-dataset-student student real student-path)
              (multiple-value-bind (native-loss native-gradients)
                  (tb:distillation-batch-loss-and-gradients
                   student native-batch :temperature 2.0 :hard-weight 0.3)
                (unwind-protect
                     (multiple-value-bind (python-loss python-gradients)
                         (tb:distillation-batch-loss-and-gradients
                          student python-batch :temperature 2.0 :hard-weight 0.3)
                       (unwind-protect
                            (progn
                              (check (< (abs (- native-loss python-loss)) tolerance)
                                     "Python/native dataset losses agree")
                              (check (= (length native-gradients) (length python-gradients))
                                     "Python/native dataset gradient coverage agrees")
                              (loop for left in native-gradients for right in python-gradients do
                                (check (equal (car left) (car right))
                                       "Python/native dataset gradient names agree")
                                (close-arrays (tb:tensor-array (cdr left))
                                              (tb:tensor-array (cdr right)) tolerance 3e-4)))
                         (mapc (lambda (entry) (tb:dispose (cdr entry))) python-gradients)))
                  (mapc (lambda (entry) (tb:dispose (cdr entry))) native-gradients))))))))))

(defun check-identical-datasets (left-directory right-directory device)
  (let ((left (tb:load-distillation-dataset left-directory :device device))
        (right (tb:load-distillation-dataset right-directory :device device)))
    (check (= (tb:distillation-dataset-size left) (tb:distillation-dataset-size right))
           "streamed and materialized dataset sizes agree")
    (dotimes (index (tb:distillation-dataset-size left))
      (tb:with-resource (left-batch (tb:load-distillation-dataset-batch left index))
        (tb:with-resource (right-batch (tb:load-distillation-dataset-batch right index))
          (check (equalp (tb::distillation-batch-ids left-batch)
                         (tb::distillation-batch-ids right-batch))
                 "streamed input snapshots agree")
          (check (equalp (tb::distillation-batch-labels left-batch)
                         (tb::distillation-batch-labels right-batch))
                 "streamed label snapshots agree")
          (check (equalp (tb::distillation-batch-mask left-batch)
                         (tb::distillation-batch-mask right-batch))
                 "streamed mask snapshots agree")
          (check (eq (tb:distillation-batch-representation left-batch)
                     (tb:distillation-batch-representation right-batch))
                 "streamed and materialized representations agree")
          (if (eq (tb:distillation-batch-representation left-batch) :dense)
              (close-arrays (tb:tensor-array (tb::distillation-batch-logits left-batch))
                            (tb:tensor-array (tb::distillation-batch-logits right-batch)) 0 0)
              (progn
                (check (= (tb:distillation-batch-top-k left-batch)
                          (tb:distillation-batch-top-k right-batch))
                       "streamed and materialized top-k values agree")
                (check (= (tb:distillation-batch-temperature left-batch)
                          (tb:distillation-batch-temperature right-batch))
                       "streamed and materialized temperatures agree")
                (close-arrays
                 (tb:tensor-array (tb::distillation-batch-top-log-probs left-batch))
                 (tb:tensor-array (tb::distillation-batch-top-log-probs right-batch)) 0 0)
                (check (equalp
                        (tb::tensor-int-array (tb::distillation-batch-top-indices left-batch))
                        (tb::tensor-int-array (tb::distillation-batch-top-indices right-batch)))
                       "streamed and materialized top-k indices agree")
                (close-arrays
                 (tb:tensor-array (tb::distillation-batch-tail-log-prob left-batch))
                 (tb:tensor-array (tb::distillation-batch-tail-log-prob right-batch)) 0 0))))))))

(defun check-dataset-manifest-rejections (directory device)
  (let* ((path (merge-pathnames "distillation-dataset.json" directory))
         (original (uiop:read-file-string path)))
    (unwind-protect
         (dolist (mutation
                   (list (lambda (manifest) (setf (gethash "format_version" manifest) 3))
                         (lambda (manifest) (setf (gethash "dataset_id" manifest) ""))
                         (lambda (manifest) (setf (gethash "batch_count" manifest) 0))
                         (lambda (manifest)
                           (setf (gethash "path" (aref (gethash "batches" manifest) 0)) "../escape"))
                         (lambda (manifest)
                           (incf (gethash "selected_positions"
                                         (aref (gethash "batches" manifest) 0))))
                         (lambda (manifest)
                           (setf (gethash "manifest_sha256"
                                          (aref (gethash "batches" manifest) 0))
                                 (make-string 64 :initial-element #\0)))
                         (lambda (manifest)
                           (setf (gethash "content_sha256" manifest)
                                 (make-string 64 :initial-element #\0)))))
           (with-open-file (stream path :direction :output :if-exists :supersede
                                        :external-format :utf-8)
             (write-string original stream))
           (let ((manifest (tb::read-json path)))
             (funcall mutation manifest)
             (tb::write-json manifest path)
             (signals tb:compatibility-error
               (tb:load-distillation-dataset directory :device device))))
      (with-open-file (stream path :direction :output :if-exists :supersede
                                   :external-format :utf-8)
        (write-string original stream)))))

(defun check-dataset-state-rejections (dataset state-directory)
  (let* ((path (merge-pathnames "dataset-state.json" state-directory))
         (original (uiop:read-file-string path))
         (position (tb:distillation-dataset-position dataset)))
    (unwind-protect
         (dolist (mutation
                   (list (lambda (manifest) (setf (gethash "format_version" manifest) 3))
                         (lambda (manifest) (setf (gethash "dataset_id" manifest) "other"))
                         (lambda (manifest) (setf (gethash "epoch" manifest) -1))
                         (lambda (manifest) (setf (gethash "position" manifest) 4))
                         (lambda (manifest)
                           (setf (gethash "content_sha256" manifest)
                                 (make-string 64 :initial-element #\0)))))
           (with-open-file (stream path :direction :output :if-exists :supersede
                                        :external-format :utf-8)
             (write-string original stream))
           (let ((manifest (tb::read-json path)))
             (funcall mutation manifest)
             (tb::write-json manifest path)
             (signals tb:compatibility-error
               (tb:restore-distillation-dataset-state dataset state-directory))
             (check (= position (tb:distillation-dataset-position dataset))
                    "rejected state leaves iterator unchanged")))
      (with-open-file (stream path :direction :output :if-exists :supersede
                                   :external-format :utf-8)
        (write-string original stream)))
    (let ((sentinel original)
          (tb::*before-directory-publication*
            (lambda (stage destination)
              (declare (ignore stage destination))
              (error "injected state failure"))))
      (signals error (tb:save-distillation-dataset-state dataset state-directory))
      (check (equal sentinel (uiop:read-file-string path))
             "failed state publication preserves prior iterator checkpoint"))))

(defun check-dataset-corruption-rejections (directory device)
  (let* ((weights (merge-pathnames "batches/000000/teacher.safetensors" directory))
         (metadata (merge-pathnames "batches/000000/distillation.json" directory))
         (weights-original (read-file-octets weights))
         (metadata-original (uiop:read-file-string metadata))
         (dataset (tb:load-distillation-dataset directory :device device)))
    (unwind-protect
         (progn
           (let ((corrupt (copy-seq weights-original)))
             (setf (aref corrupt (1- (length corrupt)))
                   (logxor #xff (aref corrupt (1- (length corrupt)))))
             (write-file-octets weights corrupt))
           (signals tb:compatibility-error
             (tb:load-distillation-dataset directory :device device))
           (signals tb:compatibility-error
             (tb:load-distillation-dataset-batch dataset 0))
           (check (= 0 (tb:distillation-dataset-position dataset))
                  "corrupt lazy batch leaves iterator unchanged"))
      (write-file-octets weights weights-original))
    (unwind-protect
         (progn
           (with-open-file (stream metadata :direction :output :if-exists :append
                                            :external-format :utf-8)
             (write-char #\Newline stream))
           (signals tb:compatibility-error
             (tb:load-distillation-dataset directory :device device))
           (signals tb:compatibility-error
             (tb:load-distillation-dataset-batch dataset 0)))
      (with-open-file (stream metadata :direction :output :if-exists :supersede
                                       :external-format :utf-8)
        (write-string metadata-original stream)))))

(defun check-legacy-dataset-v1 (directory device state-directory)
  (let* ((path (merge-pathnames "distillation-dataset.json" directory))
         (original (uiop:read-file-string path)))
    (unwind-protect
         (let ((manifest (tb::read-json path)))
           (setf (gethash "format_version" manifest) 1)
           (remhash "content_sha256" manifest)
           (loop for entry across (gethash "batches" manifest) do
             (remhash "manifest_sha256" entry)
             (remhash "weights_sha256" entry))
           (tb::write-json manifest path)
           (let ((dataset (tb:load-distillation-dataset directory :device device)))
             (check (null (tb:distillation-dataset-content-sha256 dataset))
                    "legacy dataset explicitly has no verified content identity")
             (tb:with-resource (batch (tb:load-distillation-dataset-batch dataset 0))
               (check (typep batch 'tb:distillation-batch) "legacy dataset remains readable"))
             (uiop:delete-directory-tree state-directory :validate t :if-does-not-exist :ignore)
             (tb:save-distillation-dataset-state dataset state-directory)
             (let ((state (tb::read-json (merge-pathnames "dataset-state.json" state-directory))))
               (check (= (gethash "format_version" state) 1) "legacy iterator state remains v1")
               (check (null (gethash "content_sha256" state))
                      "legacy state does not invent content identity"))
             (tb:restore-distillation-dataset-state dataset state-directory)))
      (with-open-file (stream path :direction :output :if-exists :supersede
                                   :external-format :utf-8)
        (write-string original stream)))))

(defun run-distillation-dataset-tests ()
  (check-sha256-vectors)
  (tb::ensure-native)
  (let* ((device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
         (real (equal (uiop:getenv "TB_DISTILLATION_DATASET_REAL") "1"))
         (root (asdf:system-source-directory "cl-transformer-blocks"))
         (teacher-path (if real (merge-pathnames ".build/models/smollm2/" root)
                           (merge-pathnames (format nil ".build/components-created-~(~A~)/" device)
                                            root)))
         (student-path (merge-pathnames
                        (if real (format nil ".build/recomposition-~(~A~)/real/" device)
                            (format nil ".build/recomposition-~(~A~)/untied/drop/" device)) root))
         (output-root (merge-pathnames
                       (format nil ".build/distillation-datasets-~(~A~)/~A" device
                               (if real "real/" "")) root))
         (output (merge-pathnames "native/" output-root))
         (half-output (merge-pathnames "native-fp16/" output-root))
         (streamed-output (merge-pathnames "native-streamed/" output-root))
         (streamed-half-output (merge-pathnames "native-streamed-fp16/" output-root))
         (top-half-output (merge-pathnames "native-topk-fp16/" output-root))
         (streamed-top-half-output
           (merge-pathnames "native-streamed-topk-fp16/" output-root))
         (python-output (merge-pathnames "python/" output-root))
         (python-half-output (merge-pathnames "python-fp16/" output-root))
         (python-top-half-output (merge-pathnames "python-topk-fp16/" output-root))
         (state-output (merge-pathnames
                        "state/" output-root))
         (checkpoint (merge-pathnames
                      "checkpoint/" output-root))
         (updated (merge-pathnames "native-updated/" output-root))
         (tolerance (if real 1e-3 2e-6))
         (half-tolerance (if real 1e-2 1e-3))
         (top-k (if real 64 4))
         (handles (tb::%tb-live-tensors)) (targets nil) (top-targets nil))
    (tb:with-resource (teacher (tb:from-pretrained teacher-path :device device))
      (check (member :distillation-datasets
                     (getf (tb:model-capabilities teacher) :operations))
             "native teacher advertises distillation datasets")
      (signals tb:shape-error
        (tb:make-distillation-example #2A((1))))
      (signals tb:shape-error
        (tb:make-distillation-example #2A((1 2)) :labels #2A((-100 "bad"))))
      (let* ((examples (make-streaming-examples))
             (baseline (tb::%tb-live-tensors))
             (version (tb::model-version teacher))
             (yielded 0)
             (sentinel (merge-pathnames "sentinel" streamed-output)))
        (ensure-directories-exist sentinel)
        (with-open-file (stream sentinel :direction :output :if-exists :supersede)
          (write-string "old-stream" stream))
        (signals error
          (tb:save-distillation-dataset-from-teacher
           teacher
           (make-instance
            'test-distillation-source :examples examples
            :after-yield (lambda ()
                           (check (= baseline (tb::%tb-live-tensors))
                                  "streaming releases each target before requesting another")
                           (when (= (incf yielded) 2) (error "injected source failure"))))
           streamed-output :dataset-id "tiny-three-v1"))
        (check (equal (uiop:read-file-string sentinel) "old-stream")
               "failed streaming publication preserves old dataset")
        (setf yielded 0)
        (tb:save-distillation-dataset-from-teacher
         teacher
         (make-instance
          'test-distillation-source :examples examples
          :after-yield (lambda ()
                         (incf yielded)
                         (check (= baseline (tb::%tb-live-tensors))
                                "streaming target lifetime is bounded to one visitor call")))
         streamed-output :dataset-id "tiny-three-v1")
        (check (= yielded 3) "custom source lazily yields every example")
        (check (= version (tb::model-version teacher)) "streaming keeps teacher version unchanged")
        (tb:save-distillation-dataset-from-teacher
         teacher examples streamed-half-output :dataset-id "tiny-three-v1"
         :storage-dtype :float16)
        (tb:save-distillation-dataset-from-teacher
         teacher examples streamed-top-half-output :dataset-id "tiny-three-v1"
         :storage-dtype :float16 :top-k top-k :temperature 2.0)
        (let* ((manifest-path (merge-pathnames "distillation-dataset.json" streamed-output))
               (manifest (uiop:read-file-string manifest-path)))
          (signals tb:shape-error
            (tb:save-distillation-dataset-from-teacher
             teacher #() streamed-output :dataset-id "tiny-three-v1"))
          (signals tb:compatibility-error
            (tb:save-distillation-dataset-from-teacher
             teacher #(42) streamed-output :dataset-id "tiny-three-v1"))
          (signals tb:shape-error
            (tb:save-distillation-dataset-from-teacher
             teacher examples streamed-output :dataset-id "tiny-three-v1"
             :storage-dtype :bfloat16))
          (check (equal manifest (uiop:read-file-string manifest-path))
                 "invalid streaming sources preserve the published dataset")))
      (setf targets (make-dataset-targets teacher)
            top-targets (make-dataset-targets teacher :top-k top-k :temperature 2.0)))
    (unwind-protect
         (progn
           (let ((sentinel (merge-pathnames "sentinel" output)))
             (ensure-directories-exist sentinel)
             (with-open-file (stream sentinel :direction :output :if-exists :supersede)
               (write-string "old" stream))
             (let ((tb::*before-directory-publication*
                     (lambda (stage destination)
                       (declare (ignore stage destination))
                       (error "injected dataset failure"))))
               (signals error
                 (tb:save-distillation-dataset targets output :dataset-id "tiny-three-v1")))
             (check (equal (uiop:read-file-string sentinel) "old")
                    "failed dataset publication preserves old artifact"))
           (tb:save-distillation-dataset targets output :dataset-id "tiny-three-v1")
           (tb:save-distillation-dataset targets half-output :dataset-id "tiny-three-v1"
                                         :storage-dtype :float16)
           (tb:save-distillation-dataset top-targets top-half-output
                                         :dataset-id "tiny-three-v1"
                                         :storage-dtype :float16)
           (signals tb:shape-error
             (tb:save-distillation-dataset targets half-output :dataset-id "tiny-three-v1"
                                           :storage-dtype :bfloat16)))
      (dispose-targets targets)
      (dispose-targets top-targets))
    (let ((dataset (tb:load-distillation-dataset half-output :device device
                                                :shuffle t :seed 41)))
      (check (= handles (tb::%tb-live-tensors)) "opening a dataset keeps all logits lazy")
      (check (equal (tb:distillation-dataset-id dataset) "tiny-three-v1") "dataset ID round trips")
      (check (and (stringp (tb:distillation-dataset-content-sha256 dataset))
                  (= 64 (length (tb:distillation-dataset-content-sha256 dataset))))
             "dataset exposes its verified SHA-256 content identity")
      (check (= (tb:distillation-dataset-size dataset) 3) "dataset size round trips")
      (check (= (tb:distillation-dataset-epoch dataset) 0) "dataset starts at epoch zero")
      (check (= (tb:distillation-dataset-position dataset) 0) "dataset starts at position zero")
      (tb:start-distillation-dataset-epoch dataset 2)
      (check (= (tb:distillation-dataset-epoch dataset) 2) "explicit epoch is recorded")
      (multiple-value-bind (batch index) (tb:next-distillation-dataset-batch dataset)
        (unwind-protect
             (progn (check (typep batch 'tb:distillation-batch) "next returns an owned batch")
                    (check (= index 2) "portable shuffle returns its specified first index"))
          (tb:dispose batch)))
      (check (= (tb:distillation-dataset-position dataset) 1) "successful load advances cursor")
      (tb:save-distillation-dataset-state dataset state-output)
      (check-dataset-state-rejections dataset state-output)
      (let ((restored (tb:load-distillation-dataset half-output :device device
                                                   :shuffle t :seed 41)))
        (tb:restore-distillation-dataset-state restored state-output)
        (check (= (tb:distillation-dataset-epoch restored) 2) "state restores epoch")
        (check (= (tb:distillation-dataset-position restored) 1) "state restores next position")
        (loop for expected in '(1 0) do
          (multiple-value-bind (batch index) (tb:next-distillation-dataset-batch restored)
            (unwind-protect (check (= index expected) "restored shuffle order agrees")
              (tb:dispose batch))))
        (check (null (tb:next-distillation-dataset-batch restored)) "iterator ends explicitly"))
      (let ((wrong (tb:load-distillation-dataset half-output :device device
                                                :shuffle t :seed 42)))
        (signals tb:compatibility-error
          (tb:restore-distillation-dataset-state wrong state-output)))
      (let ((wrong-content (tb:load-distillation-dataset output :device device
                                                        :shuffle t :seed 41)))
        (signals tb:compatibility-error
          (tb:restore-distillation-dataset-state wrong-content state-output)))
      (tb:start-distillation-dataset-epoch dataset 3)
      (tb:with-resource (student (tb:from-pretrained student-path :device device))
        (activate-dataset-student student real student-path)
        (tb:with-resource (optimizer (tb:make-sgd :learning-rate 0.01 :momentum 0.9
                                                  :weight-decay 0.02))
          (let ((position (tb:distillation-dataset-position dataset)))
            (signals tb:shape-error
              (tb:distill-dataset-step student dataset optimizer :hard-weight 2))
            (check (= position (tb:distillation-dataset-position dataset))
                   "failed update does not consume a batch"))
          (check (realp (tb:distill-dataset-step
                         student dataset optimizer :temperature 2.0 :hard-weight 0.3
                         :max-grad-norm 0.05))
                 "dataset performs a native optimizer step")
          (tb:save-training-checkpoint student optimizer checkpoint)
          (tb:save-distillation-dataset-state dataset state-output)
          (let ((remaining-losses (drain-dataset student dataset optimizer))
                (expected nil))
            (setf expected (parameter-arrays student))
            (if real (tb:save-adapter student updated) (tb:save-pretrained student updated))
            (tb:with-resource (resumed (tb:from-pretrained (if real student-path checkpoint)
                                                            :device device))
              (when real (tb:load-adapter resumed checkpoint))
              (tb:with-resource (resumed-optimizer
                                  (tb:make-sgd :learning-rate 0.01 :momentum 0.9
                                               :weight-decay 0.02))
                (tb:restore-training-checkpoint resumed resumed-optimizer checkpoint)
                (let ((resumed-dataset
                        (tb:load-distillation-dataset half-output :device device
                                                     :shuffle t :seed 41)))
                  (tb:restore-distillation-dataset-state resumed-dataset state-output)
                  (check (equal remaining-losses
                                (drain-dataset resumed resumed-dataset resumed-optimizer))
                         "resumed dataset losses are exact")
                  (check-parameter-arrays resumed expected))))))))
    (check-python-dataset output python-output student-path device real tolerance :float32)
    (check-python-dataset half-output python-half-output student-path device real
                          half-tolerance :float16)
    (check-python-dataset top-half-output python-top-half-output student-path device real
                          half-tolerance :float16)
    (check-identical-datasets output streamed-output device)
    (check-identical-datasets half-output streamed-half-output device)
    (check-identical-datasets top-half-output streamed-top-half-output device)
    (check-dataset-manifest-rejections output device)
    (check-dataset-corruption-rejections output device)
    (check-legacy-dataset-v1 output device (merge-pathnames "legacy-state/" output-root))
    (check (= handles (tb::%tb-live-tensors)) "dataset iteration returns native handles to baseline")
    (format t "~&Distillation dataset iteration passed: ~A, real=~S (~D checks).~%"
            device real *checks*)))
