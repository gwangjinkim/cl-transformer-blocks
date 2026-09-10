(load "scripts/load.lisp")

(defun dispose-outputs (outputs)
  (mapc (lambda (entry) (tb:dispose (cdr entry))) outputs))

(defun median (values)
  (let* ((sorted (sort (copy-list values) #'<)) (count (length sorted)))
    (if (oddp count) (nth (floor count 2) sorted)
        (/ (+ (nth (1- (/ count 2)) sorted) (nth (/ count 2) sorted)) 2.0))))

(let* ((root (asdf:system-source-directory "cl-transformer-blocks"))
       (fixture (merge-pathnames ".build/fixtures/python-worker-opt/" root))
       (device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
       (embeds (make-array '(8 64 16) :element-type 'single-float))
       (mask (make-array '(8 64) :initial-element 1))
       (units internal-time-units-per-second))
  (dotimes (index (array-total-size embeds))
    (setf (row-major-aref embeds index)
          (coerce (/ (- (mod index 31) 15) 100.0) 'single-float)))
  (tb:with-resource
      (model (tb:from-pretrained fixture :execution :python :device device
                                 :auto-class "AutoModelForCausalLM"
                                 :local-files-only t :max-output-elements 20000
                                 :python-threads 1))
    (labels ((call (transport)
               (dispose-outputs
                (tb:python-forward
                 model `(("inputs_embeds" . ,embeds) ("attention_mask" . ,mask))
                 :transport transport)))
             (measure (transport)
               (loop repeat 9 collect
                 (let ((start (get-internal-real-time)))
                   (call transport)
                   (/ (- (get-internal-real-time) start) units 1.0)))))
      (dotimes (index 2) (declare (ignore index)) (call :json) (call :binary))
      (let ((json (median (measure :json))) (binary (median (measure :binary))))
        (format t "device=~(~A~) elements=16384 json_seconds=~,6F binary_seconds=~,6F ratio=~,3F~%"
                device json binary (/ binary json))))))
