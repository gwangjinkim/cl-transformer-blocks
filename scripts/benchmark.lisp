(load "scripts/load.lisp")

(let* ((device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
       (n (parse-integer (or (uiop:getenv "TB_BENCH_N") "512")))
       (iterations (parse-integer (or (uiop:getenv "TB_BENCH_ITERATIONS") "40")))
       (repeats (parse-integer (or (uiop:getenv "TB_BENCH_REPEATS") "5")))
       (result (make-hash-table :test 'equal)))
  (unless (and (<= 1 n 4096) (<= 1 iterations 100000) (<= 1 repeats 100))
    (error "Invalid benchmark dimensions/counts"))
  (tb:with-resource (backend (tb:make-backend :device device))
    (let ((tb::*backend* backend))
      (tb:with-resource
          (a (tb:tensor-from-array
              backend (make-array (list n n) :element-type 'single-float
                                            :initial-element (/ 1.0 n))))
        (flet ((step-once ()
                 (tb::with-tensor-scope
                   (tb::checked-status
                    (tb::%tb-tensor-eval (tb::pointer (tb::matmul a a)))))))
          (dotimes (i 10) (step-once))
          (setf (gethash "seconds" result)
                (coerce
                 (loop repeat repeats collect
                   (let ((start (get-internal-real-time)))
                     (dotimes (i iterations) (step-once))
                     (/ (- (get-internal-real-time) start)
                        (float (* iterations internal-time-units-per-second) 1.0d0))))
                 'vector)))
        ;; Validate outside the timed loop and report the actual native readback.
        (tb::with-tensor-scope
          (let ((values (tb:tensor-array (tb::matmul a a))))
            (dotimes (i (array-total-size values))
              (assert (< (abs (- (row-major-aref values i) (/ 1.0 n))) 1e-6)))
            (setf (gethash "verified_value" result) (row-major-aref values 0)))))))
  (setf (gethash "frontend" result) "lisp"
        (gethash "device" result) (string-downcase device)
        (gethash "dimension" result) n
        (gethash "iterations" result) iterations)
  (yason:encode result *standard-output*)
  (terpri))
