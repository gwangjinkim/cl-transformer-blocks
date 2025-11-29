(in-package #:cl-transformer-blocks-mgl)

;;; ------------------------------------------------------------
;;; Helpers
;;; ------------------------------------------------------------

(defun %ensure-mat (x)
  (assert (typep x 'mgl-mat:mat))
  x)

;;; ------------------------------------------------------------
;;; Zero / shape
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:tb-zeros ((backend (eql :mgl)) dims &key (dtype :float))
  "Create an MGL-MAT:MAT of zeros with shape DIMS."
  (declare (ignore dtype))
  (apply #'mgl-mat:zeros dims))

(defmethod cl-transformer-blocks:tb-tensor-shape ((x mgl-mat:mat))
  "Return dimensions of an MGL-MAT:MAT as a list."
  (mgl-mat:dimensions x))

;;; ------------------------------------------------------------
;;; Basic ops
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:tb-matmul ((a mgl-mat:mat) (b mgl-mat:mat))
  (mgl-mat:matmul a b))

(defmethod cl-transformer-blocks:tb-add ((a mgl-mat:mat) (b mgl-mat:mat))
  (mgl-mat:+ a b))

(defmethod cl-transformer-blocks:tb-add-scaled ((a mgl-mat:mat) (b mgl-mat:mat) scale)
  ;; a + scale * b
  (mgl-mat:+ a (mgl-mat:* scale b)))

(defmethod cl-transformer-blocks:tb-scale ((x mgl-mat:mat) alpha)
  (mgl-mat:* alpha x))

(defmethod cl-transformer-blocks:tb-transpose ((x mgl-mat:mat))
  (mgl-mat:transpose x))

;;; ------------------------------------------------------------
;;; Softmax
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:tb-softmax ((x mgl-mat:mat) &key axis)
  (declare (ignore axis))
  ;; We rely on MGL's softmax; for now treat as "softmax over last axis".
  (mgl-mat:softmax x))

;;; ------------------------------------------------------------
;;; GELU
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:tb-gelu ((x mgl-mat:mat))
  "Approximate GELU using tanh-based formula."
  (let* ((c0 0.5d0)
         (c1 (* (sqrt 2d0 (/ 1d0 pi)) 0.5d0))) ; 0.5 * sqrt(2/pi)
    (mgl-mat:with-cuda* ()
      (let ((x3 (mgl-mat:* x (mgl-mat:* x x))) ; x^3
            (inner nil)
            (tanh-inner nil)
            (one (mgl-mat:ones-like x)))
        (setf inner (mgl-mat:+ x (mgl-mat:* 0.044715d0 x3)))
        (setf tanh-inner (mgl-mat:tanh (mgl-mat:* c1 inner)))
        ;; 0.5 * x * (1 + tanh(...))
        (mgl-mat:* c0 x (mgl-mat:+ one tanh-inner))))))
        
;;; ------------------------------------------------------------
;;; Layer norm
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:tb-layer-norm
    ((x mgl-mat:mat) gamma beta &key (epsilon 1.0d-5))
  "Simple layer norm over features for each column.

GAMMA/BETA can be NIL, in which case we use scale=1 and shift=0."
  (declare (ignore gamma beta))
  (mgl-mat:with-cuda* ()
    (let* ((dims (mgl-mat:dimensions x))
           (rows (first dims))
           (cols (second dims))
           ;; mean & variance per column
           (mean (mgl-mat:zeros rows 1))
           (var  (mgl-mat:zeros rows 1))
           (ones (mgl-mat:ones 1 cols))
           (normed (mgl-mat:zeros rows cols)))
      ;; mean across columns: mean = (1/C) * X * 1
      (setf mean (mgl-mat:matmul x ones))
      (setf mean (mgl-mat:* (/ 1.0d0 cols) mean))

      ;; variance: mean of squared deviation
      (let* ((mean-broadcast (mgl-mat:matmul mean (mgl-mat:ones 1 cols)))
             (centered (mgl-mat:- x mean-broadcast))
             (sq (mgl-mat:* centered centered)))
        (setf var (mgl-mat:matmul sq ones))
        (setf var (mgl-mat:* (/ 1.0d0 cols) var))
        ;; stddev
        (let* ((var-bc (mgl-mat:matmul var (mgl-mat:ones 1 cols)))
               (denom (mgl-mat:sqrt (mgl-mat:+ var-bc epsilon)))
               (y (mgl-mat:/ centered denom)))
          (setf normed y)))
      normed)))

;;; ------------------------------------------------------------
;;; Dropout
;;; ------------------------------------------------------------

(defmethod cl-transformer-blocks:tb-dropout
    ((x mgl-mat:mat) p &key training-p)
  "Naive dropout implementation for MGL-MAT backend."
  (cond
    ;; no dropout
    ((or (not training-p)
         (<= p 0.0d0))
     (mgl-mat:copy! x))
    ;; extreme: everything dropped -> zeros
    ((>= p 1.0d0)
     (apply #'mgl-mat:zeros (mgl-mat:dimensions x)))
    (t
     (let* ((dims (mgl-mat:dimensions x))
            (rows (first dims))
            (cols (second dims))
            (y    (mgl-mat:copy! x))
            (mask (apply #'mgl-mat:zeros dims)))
       ;; fill mask with uniform [0,1)
       (mgl-mat:uniform-random! mask :limit 1.0d0)
       (let ((scale (/ 1.0d0 (- 1.0d0 p))))
         (dotimes (i rows)
           (dotimes (j cols)
             (let ((r (mgl-mat:mref mask i j)))
               (setf (mgl-mat:mref mask i j)
                     (if (< r p)
                         0.0d0
                         scale))))))
       ;; apply mask elementwise
       (dotimes (i rows)
         (dotimes (j cols)
           (setf (mgl-mat:mref y i j)
                 (* (mgl-mat:mref y i j)
                    (mgl-mat:mref mask i j)))))
       y))))
