(in-package #:cl-transformer-blocks-mgl)

;;; --------------------
;;; protocol: allocation
;;; --------------------

(defmethod tb-zeros ((backend (eql :mgl)) dims &key (dtype :float))
  "Create a zero-initialized MGL-MAT:MAT of shape DIMS.
  
BACKEND is ignored except for dispatch; it should be :MGL.
DTYPE is passed to MGL-MAT:MAKE-MAT as CTYPE when it makes sense."
  (declare (ignore backend))
  ;; DIMS is e.g. (rows cols) or (d t b)
  (mgl-mat:make-mat dims :ctype dtype :initial-element 0.0d0))


;;; --------------------
;;; basic allocation
;;; --------------------


(defun random-mat (rows cols &key (stddev 0.02d0))
  "Create a ROWS x COLS MGL-MAT:MAT with small random values in [-SCALE, SCALE]."
  (let ((m (tb:tb-zeros :mgl (list rows cols))))
    (mgl-mat:gaussian-random! m :mean 0 :stddev stddev)
    m))

;; if you really want a uniform in [-sacle, scale], then do:

;; (defun random-mat (rows cols &key (scale 0.02d0))
;;   "Create a ROWS x COLS MGL-MAT:MAT with small random values in [-SCALE, SCALE]."
;;   (let ((m (tb:tb-zeros :mgl (list rows cols))))
;;     (mgl-mat:uniform-random! m :limit (* 2.0d0 scale))
;;     ;; Shift to [-scale, scale).
;;     (mgl-mat:.+! (- scale) m)
;;     m))



;;; --------------------
;;; protocol: basic ops
;;; --------------------

;;; Implement the protocol generics for MGL-MAT:MAT

(defmethod tb-tensor-shape ((x mat))
  (mat-dimensions x))

(defmethod tb-matmul ((a mat) (b mat))
  "Matrix multiplication via GEMM (BLAS/cuBLAS)."
  (mm* a b))

(defmethod tb-add ((a mat) (b mat))
  "Return a + b as a new MAT."
  (let ((y (copy-mat a)))
    (axpy! 1.0d0 b y)  ; y := y + 1.0 * b
    y))

(defmethod tb-add-scaled ((a mat) (b mat) scale)
  "Return a + scale * b as a new MAT."
  (let ((y (copy-mat a)))
    (axpy! scale b y)
    y))

(defmethod tb-scale ((x mat) alpha)
  "Return alpha * x as a new MAT."
  (let ((y (copy-mat x)))
    (scal! alpha y)
    y))

(defmethod tb-transpose ((x mat))
  (transpose x))


;; ;;; --------------------
;; ;;; protocol: GELU
;; ;;; --------------------

;; (defmethod tb-gelu ((x mat))
;;   "Elementwise approximate GELU on MAT."
;;   (let* ((dims (mat-dimensions x))
;;          (rows (first dims))
;;          (cols (second dims))
;;          (y    (copy-mat x)))
;;     (dotimes (i rows)
;;       (dotimes (j cols)
;;         (let* ((v (mref y i j))
;;                (u (/ v (sqrt 2.0d0)))
;;                (g (* 0.5d0 v (+ 1.0d0 (erf u)))))
;;           (setf (mref y i j) g))))
;;     y))

;;; ------------------------------------------------------------
;;; GELU (approximate, no ERF needed)
;;; ------------------------------------------------------------

(defun %gelu-approx (x)
  "Approximate GELU using tanh-based formula (no ERF)."
  ;; Hendrycks & Gimpel approximation:
  ;; GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 x^3)))
  (let* ((c (/ (sqrt (* 2.0d0 pi))))           ; √(2/π)
         (x3 (* x x x))
         (inner (+ x (* 0.044715d0 x3)))
         (tanh-arg (* c inner)))
    (* 0.5d0 x (+ 1.0d0 (tanh tanh-arg)))))

(defmethod tb-gelu ((x mat))
  "Elementwise approximate GELU on MAT."
  (let* ((dims (mat-dimensions x))
         (rows (first dims))
         (cols (second dims))
         (y    (copy-mat x)))
    (dotimes (i rows)
      (dotimes (j cols)
        (let ((v (mref y i j)))
          (setf (mref y i j) (%gelu-approx v)))))
    y))

;;; --------------------
;;; protocol: softmax (2D, axis=-1)
;;; --------------------

(defun %softmax-row! (m row)
  "In-place softmax of ROW of M (2D MAT) using numerical stabilization."
  (let* ((dims (mat-dimensions m))
         (cols (second dims)))
    ;; 1. max for numerical stability
    (let ((max-val -1d300))
      (dotimes (j cols)
        (let ((v (mref m row j)))
          (when (> v max-val)
            (setf max-val v))))
      ;; 2. exponentiate shifted values & accumulate sum
      (let ((sum 0d0))
        (dotimes (j cols)
          (let* ((v  (mref m row j))
                 (ev (exp (- v max-val))))
            (setf (mref m row j) ev)
            (incf sum ev)))
        ;; 3. normalize
        (dotimes (j cols)
          (setf (mref m row j)
                (/ (mref m row j) sum)))))))

(defmethod tb-softmax ((x mat) &key (axis -1))
  "Softmax over the last axis for 2D MATS. Returns a new MAT."
  (unless (eql axis -1)
    (error "tb-softmax (mgl): only AXIS = -1 is supported right now, got ~S"
           axis))
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-softmax (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (y    (copy-mat x)))
      (dotimes (i rows)
        (%softmax-row! y i))
      y)))


;;; --------------------
;;; protocol: layer-norm
;;; --------------------

;;; This implementation only uses mat-dimensions, mref, copy-mat, make-mat (indirectly),
;;; and has no mysterious MGL internals,
;;; and it's correct enough for now. (later optimization with custom kernels possible).


(defun %layer-norm-row! (x out row gamma beta eps)
  "Compute layer-norm for one ROW of X into OUT.

X, OUT are MGL-MAT:MATS of shape (rows x cols).
GAMMA, BETA are either NIL or MATS of shape (1 x cols).
EPS is a small float for numerical stability."
  (let* ((dims (mat-dimensions x))
         (cols (second dims)))
    ;; 1. compute mean and variance for this row
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
        ;; 2. normalize and apply gamma/beta
        (dotimes (j cols)
          (let* ((v (mref x row j))
                 (norm (* (- v mean) inv-std))
                 (g (if gamma
                        (mref gamma 0 j)
                        1.0d0))
                 (b (if beta
                        (mref beta 0 j)
                        0.0d0))
                 (y (+ (* norm g) b)))
            (setf (mref out row j) y)))))))

(defmethod tb-layer-norm ((x mat) gamma beta &key (eps 1e-5))
  "Layer normalization over the last dimension (features) for 2D MATS.

X is (rows x cols). GAMMA and BETA are either NIL or (1 x cols) MATS
(broadcast across rows)."
  (let* ((dims (mat-dimensions x)))
    (unless (= (length dims) 2)
      (error "tb-layer-norm (mgl): only 2D MATS supported, got dims ~S" dims))
    (let* ((rows (first dims))
           (cols (second dims))
           ;; sanity check gamma/beta if provided
           (gamma* (when gamma
                     (let ((gdims (mat-dimensions gamma)))
                       (unless (and (= (length gdims) 2)
                                    (= (first gdims) 1)
                                    (= (second gdims) cols))
                         (error "tb-layer-norm (mgl): GAMMA must be (1 x ~D), got ~S"
                                cols gdims))
                       gamma)))
           (beta*  (when beta
                     (let ((bdims (mat-dimensions beta)))
                       (unless (and (= (length bdims) 2)
                                    (= (first bdims) 1)
                                    (= (second bdims) cols))
                         (error "tb-layer-norm (mgl): BETA must be (1 x ~D), got ~S"
                                cols bdims))
                       beta)))
           (out (copy-mat x)))
      (dotimes (i rows)
        (%layer-norm-row! x out i gamma* beta* eps))
      out)))


;;; --------------------
;;; protocol: dropout
;;; --------------------

;;; Only uses mat-dimensions, mref, copy-mat, mgl-zeros, uniform-random!


(defmethod tb-dropout ((x mat) p &key training-p)
  "Dropout for MGL-MAT backend.

P is the drop probability in [0,1). When TRAINING-P is true, units are
zeroed with probability P and the remaining ones are scaled by 1/(1-P).
When TRAINING-P is NIL, returns a copy of X (no dropout applied)."
  (cond
    ;; no dropout when not training or p <= 0
    ((or (not training-p)
         (<= p 0.0d0))
     (copy-mat x))

    ;; extreme corner: p >= 1.0 -> all zeros
    ((>= p 1.0d0)
     (tb:tb-zeros :mgl (mat-dimensions x)))

    (t
     (let* ((dims (mat-dimensions x))
            (rows (first dims))
            (cols (second dims))
            (y    (copy-mat x))
            (mask (tb:tb-zeros :mgl dims)))
       ;; fill mask with uniform [0,1)
       (uniform-random! mask :limit 1.0d0)
       ;; threshold + scale
       (let ((scale (/ 1.0d0 (- 1.0d0 p))))
         (dotimes (i rows)
           (dotimes (j cols)
             (let ((r (mref mask i j)))
               (setf (mref mask i j)
                     (if (< r p)
                         0.0d0
                         scale))))))
       ;; apply mask: y := y * mask elementwise
       (dotimes (i rows)
         (dotimes (j cols)
           (setf (mref y i j)
                 (* (mref y i j) (mref mask i j)))))
       y))))
