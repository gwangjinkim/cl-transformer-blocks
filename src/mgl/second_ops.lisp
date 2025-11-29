;;; src/mgl/ops.lisp

(in-package #:cl-transformer-blocks-mgl)

;;; ------------------------------------------------------------
;;; allocation & helpers
;;; ------------------------------------------------------------

(defmethod tb-zeros ((backend (eql :mgl)) dims &key (dtype :float))
  "Create a zero-initialized MGL-MAT:MAT of shape DIMS.
BACKEND is ignored except for dispatch."
  (declare (ignore backend dtype))
  (make-mat dims :ctype :float :initial-element 0.0d0))

(defun random-mat (rows cols &key (stddev 0.02d0))
  "Create a ROWS x COLS MGL-MAT:MAT with small Gaussian random values."
  (let ((m (tb:tb-zeros :mgl (list rows cols))))
    (gaussian-random! m :mean 0.0d0 :stddev stddev)
    m))

;;; ------------------------------------------------------------
;;; basic ops
;;; ------------------------------------------------------------

(defmethod tb-tensor-shape ((x mat))
  (mat-dimensions x))

(defmethod tb-matmul ((a mat) (b mat))
  (mm* a b))

(defmethod tb-add ((a mat) (b mat))
  (let ((y (copy-mat a)))
    (axpy! 1.0d0 b y)
    y))

(defmethod tb-add-scaled ((a mat) (b mat) scale)
  (let ((y (copy-mat a)))
    (axpy! scale b y)
    y))

(defmethod tb-scale ((x mat) alpha)
  (let ((y (copy-mat x)))
    (scal! alpha y)
    y))

(defmethod tb-transpose ((x mat))
  (transpose x))

;;; ------------------------------------------------------------
;;; GELU (approximate)
;;; ------------------------------------------------------------

(defun %gelu-approx (x)
  "Approximate GELU using tanh-based formula."
  (let* ((c (/ (sqrt (* 2.0d0 pi))))
         (inner (+ x (* 0.044715d0 x x x)))
         (tanh-arg (* c inner)))
    (* 0.5d0 x (+ 1.0d0 (tanh tanh-arg)))))

(defmethod tb-gelu ((x mat))
  (let* ((dims (mat-dimensions x))
         (rows (first dims))
         (cols (second dims))
         (y    (copy-mat x)))
    (dotimes (i rows)
      (dotimes (j cols)
        (let ((v (mref y i j)))
          (setf (mref y i j) (%gelu-approx v)))))
    y))

;;; ------------------------------------------------------------
;;; softmax (2D, axis = -1)
;;; ------------------------------------------------------------

(defun %softmax-row! (m row)
  "In-place softmax of ROW of M (2D MAT)."
  (let* ((dims (mat-dimensions m))
         (cols (second dims)))
    ;; max for stability
    (let ((max-val -1d300))
      (dotimes (j cols)
        (let ((v (mref m row j)))
          (when (> v max-val)
            (setf max-val v))))
      ;; exp shifted + sum
      (let ((sum 0d0))
        (dotimes (j cols)
          (let* ((v  (mref m row j))
                 (ev (exp (- v max-val))))
            (setf (mref m row j) ev)
            (incf sum ev)))
        ;; normalize
        (dotimes (j cols)
          (setf (mref m row j)
                (/ (mref m row j) sum)))))))

(defmethod tb-softmax ((x mat) &key (axis -1))
  (unless (eql axis -1)
    (error "tb-softmax (mgl): only AXIS = -1 supported, got ~S" axis))
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-softmax (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (y    (copy-mat x)))
      (dotimes (i rows)
        (%softmax-row! y i))
      y)))

;;; ------------------------------------------------------------
;;; layer-norm (2D, over last dim)
;;; ------------------------------------------------------------

(defun %layer-norm-row! (x out row gamma beta eps)
  "Layer-norm for one row of X into OUT."
  (let* ((dims (mat-dimensions x))
         (cols (second dims)))
    (let ((sum 0d0)
          (sumsq 0d0))
      (dotimes (j cols)
        (let ((v (mref x row j)))
          (incf sum v)
          (incf sumsq (* v v))))
      (let* ((cols-d (coerce cols 'double-float))
             (mean   (/ sum cols-d))
             (var    (max eps (- (/ sumsq cols-d) (* mean mean))))
             (inv-std (/ 1.0d0 (sqrt var))))
        (dotimes (j cols)
          (let* ((v    (mref x row j))
                 (norm (* (- v mean) inv-std))
                 (g    (if gamma
                           (mref gamma 0 j)
                           1.0d0))
                 (b    (if beta
                           (mref beta 0 j)
                           0.0d0))
                 (yval (+ (* norm g) b)))
            (setf (mref out row j) yval)))))))

(defmethod tb-layer-norm ((x mat) gamma beta &key (eps 1e-5))
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-layer-norm (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (cols (second dims))
           (gamma* (when gamma
                     (let ((gdims (mat-dimensions gamma)))
                       (unless (and (= (length gdims) 2)
                                    (= (first gdims) 1)
                                    (= (second gdims) cols))
                         (error "GAMMA must be (1 x ~D), got ~S" cols gdims))
                       gamma))
           (beta*  (when beta
                     (let ((bdims (mat-dimensions beta)))
                       (unless (and (= (length bdims) 2)
                                    (= (first bdims) 1)
                                    (= (second bdims) cols))
                         (error "BETA must be (1 x ~D), got ~S" cols bdims))
                       beta))
           (out (copy-mat x)))
      (dotimes (i rows)
        (%layer-norm-row! x out i gamma* beta* eps))
      out)))

;;; ------------------------------------------------------------
;;; dropout
;;; ------------------------------------------------------------

(defmethod tb-dropout ((x mat) p &key training-p)
  (cond
    ((or (not training-p)
         (<= p 0.0d0))
     (copy-mat x))
    ((>= p 1.0d0)
     (tb:tb-zeros :mgl (mat-dimensions x)))
    (t
     (let* ((dims (mat-dimensions x))
            (rows (first dims))
            (cols (second dims))
            (y    (copy-mat x))
            (mask (tb:tb-zeros :mgl dims))
            (scale (/ 1.0d0 (- 1.0d0 p))))
       ;; mask ~ U[0,1)
       (uniform-random! mask :limit 1.0d0)
       ;; threshold + scale
       (dotimes (i rows)
         (dotimes (j cols)
           (let ((r (mref mask i j)))
             (setf (mref mask i j)
                   (if (< r p) 0.0d0 scale)))))
       ;; apply mask element-wise
       (dotimes (i rows)
         (dotimes (j cols)
           (setf (mref y i j)
                 (* (mref y i j) (mref mask i j)))))
       y))))
