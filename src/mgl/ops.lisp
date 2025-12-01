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
  (declare (ignore backend))
  (mgl-mat:make-mat dims :ctype dtype :initial-element 0.0d0))


(defun random-mat (rows cols &key (stddev 0.02d0))
  "Create a ROWS x COLS MGL-MAT:MAT with small Gaussian random values."
  (let ((m (tb-zeros :mgl (list rows cols))))
    (mgl-mat:gaussian-random! m :mean 0.0d0 :stddev stddev)
    m))

;;; ------------------------------------------------------------
;;; Basic ops
;;; ------------------------------------------------------------


(defmethod cl-transformer-blocks:tb-tensor-shape ((x mgl-mat:mat))
  "Return dimensions of an MGL-MAT:MAT as a list."
  (mgl-mat:mat-dimensions x))


(defmethod tb-tensor-shape ((x mat))
  "Return the shape of X as a list of integers, e.g. (ROWS COLS)."
  (mat-dimensions x))

(defmethod tb-matmul ((a mat) (b mat))
  "Matrix multiplication A * B via MGL-MAT (BLAS / cuBLAS underneath)."
  (mm* a b))

(defmethod tb-add ((a mat) (b mat))
  "Element-wise addition. Return a new MAT representing A + B."
  (let ((y (copy-mat a)))
    ;; y := 1 * b + y  => y = a + b
    (axpy! 1.0d0 b y)
    y))

(defmethod tb-add-scaled ((a mat) (b mat) scale)
  "Return A + SCALE * B as a new MAT."
  (let ((y (copy-mat a)))
    ;; y := scale * b + y  => y = a + scale * b
    (axpy! scale b y)
    y))

(defmethod tb-scale ((x mat) alpha)
  "Return ALPHA * X as a new MAT."
  (let ((y (copy-mat x)))
    (scal! alpha y)
    y))

(defmethod tb-transpose ((x mat))
  "Return the transpose of X as a new MAT."
  (transpose x))

;;; ------------------------------------------------------------
;;; Softmax
;;; ------------------------------------------------------------
;;; Very simple 2D softmax implementation over the last axis (rows-by-row).
;;; We ignore AXIS for now and always normalize across columns, which matches
;;; the (D x T) / (T x T) usage where softmax is over the 'time' dimension.

(defmethod tb-softmax ((x mat) &key axis)
  "Softmax of X along the last dimension.

Currently AXIS is ignored; we always normalize each row independently."
  (declare (ignore axis))
  (destructuring-bind (rows cols) (mat-dimensions x)
    (let ((y (copy-mat x)))
      (dotimes (i rows)
        ;; numerical stability: subtract row max
        (let ((row-max (mref y i 0)))
          (dotimes (j cols)
            (let ((v (mref y i j)))
              (when (> v row-max)
                (setf row-max v)))
            )
          ;; exponentiate shifted values and accumulate sum
          (let ((sum 0.0d0))
            (dotimes (j cols)
              (let* ((v  (- (mref y i j) row-max))
                     (ev (exp v)))
                (setf (mref y i j) ev)
                (incf sum ev))
              )
            ;; normalize
            (dotimes (j cols)
              (setf (mref y i j)
                    (/ (mref y i j) sum)))))
        )
      y)))


;;; ------------------------------------------------------------
;;; GELU activation (approximate)
;;; ------------------------------------------------------------
;;; We use the common tanh-based approximation:
;;;   gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))

(defmethod tb-gelu ((x mat))
  "Apply GELU activation element-wise using the tanh approximation."
  (destructuring-bind (rows cols) (mat-dimensions x)
    (let* ((y (copy-mat x))
           (sqrt-2/pi (sqrt (/ 2.0d0 pi))))
      (dotimes (i rows)
        (dotimes (j cols)
          (let* ((v   (mref y i j))
                 (v3  (* v v v))
                 (inner (+ v (* 0.044715d0 v3)))
                 (targ (* sqrt-2/pi inner))
                 (phi  (tanh targ))
                 (gelu (* 0.5d0 v (+ 1.0d0 phi))))
            (setf (mref y i j) gelu))))
      y)))
        
;;; ------------------------------------------------------------
;;; Layer norm
;;; ------------------------------------------------------------

;;; layer norm implementation – keep your logic, just replace ZEROS/ONES
(defmethod cl-transformer-blocks:tb-layer-norm
    ((x mgl-mat:mat) gamma beta &key (epsilon 1d-5))
  (let* ((dims  (mgl-mat:mat-dimensions x))
         (rows  (first dims))
         (cols  (second dims))
         ;; INTERNAL: per-mat ctype, e.g. :float or :double, CPU or CUDA
         (ctype (slot-value x 'mgl-mat::ctype))
         ;; row-wise mean/var -> 1 x cols
         (mean  (mgl-mat:make-mat (list 1 cols)
                                  :ctype ctype
                                  :initial-element 0.0d0))
         (var   (mgl-mat:make-mat (list 1 cols)
                                  :ctype ctype
                                  :initial-element 0.0d0)))
    ;; accumulate mean and variance for each column
    (dotimes (j cols)
      (let ((sum 0d0)
            (sqsum 0d0))
        (dotimes (i rows)
          (let ((v (coerce (mgl-mat:mref x i j) 'double-float)))
            (incf sum v)
            (incf sqsum (* v v))))
        (let* ((m (/ sum rows))
               (v (- (/ sqsum rows) (* m m))))
          (setf (mgl-mat:mref mean 0 j) m)
          (setf (mgl-mat:mref var  0 j) v)))

      ;; end dotimes j
      )

    ;; default gamma / beta if NIL – keep same ctype as X
    (let* ((gamma (or gamma
                      (let ((g (mgl-mat:make-mat (list 1 cols)
                                                 :ctype ctype
                                                 :initial-element 1.0d0)))
                        g)))
           (beta  (or beta
                      (mgl-mat:make-mat (list 1 cols)
                                        :ctype ctype
                                        :initial-element 0.0d0))))
      ;; y = (x - mean) / sqrt(var + eps)
      (let ((y (mgl-mat:copy-mat x)))
        (dotimes (i rows)
          (dotimes (j cols)
            (let* ((m    (coerce (mgl-mat:mref mean 0 j) 'double-float))
                   (v    (max 0d0 (coerce (mgl-mat:mref var  0 j) 'double-float)))
                   (den  (sqrt (+ v epsilon)))
                   (xij  (coerce (mgl-mat:mref x i j) 'double-float))
                   (norm (/ (- xij m) den)))
              (setf (mgl-mat:mref y i j) norm))))

        ;; affine transform: y * gamma + beta (broadcast along rows)
        (dotimes (i rows)
          (dotimes (j cols)
            (let* ((g (coerce (mgl-mat:mref gamma 0 j) 'double-float))
                   (b (coerce (mgl-mat:mref beta  0 j) 'double-float))
                   (v (coerce (mgl-mat:mref y i j) 'double-float)))
              (setf (mgl-mat:mref y i j) (+ (* v g) b)))))
        y))))

;; Adjust to your style – the key part is: use make-mat + fill!, not mgl-mat:zeros / mgl-mat:ones, 
;; and keep &key (epsilon ...) so it matches the generic.)


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
     (mgl-mat:copy-mat x))

    ;; extreme: everything dropped -> zeros
    ((>= p 1.0d0)
     (apply #'cl-transformer-blocks:tb-zeros (mgl-mat:mat-dimensions x)))

    (t
     (let* ((dims (mgl-mat:mat-dimensions x))
            (rows (first dims))
            (cols (second dims))
            ;; copy of x
            (y    (mgl-mat:copy-mat x))
            ;; mask with same shape, zeros
            (mask (apply #'cl-transformer-blocks:tb-zeros dims)))
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
