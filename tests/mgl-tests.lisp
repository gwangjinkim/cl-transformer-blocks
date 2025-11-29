(in-package #:cl-transformer-blocks.tests)

(in-suite :cl-transformer-blocks/mgl)

;;; ------------------------------------------------------------
;;; basic tensor tests
;;; ------------------------------------------------------------

(test zeros-shape
  (let* ((dims '(3 4))
         (x    (tb-zeros :mgl dims)))
    (is (equal dims (tb-tensor-shape x)))))

(test random-mat-shape
  (let* ((rows 5)
         (cols 7)
         (x    (random-mat rows cols)))
    (is (equal (list rows cols)
               (tb-tensor-shape x)))))

;;; ------------------------------------------------------------
;;; single block forward
;;; ------------------------------------------------------------

(test single-block-forward-shape
  (let* ((d          32)
         (time-steps 10)
         (x          (random-mat d time-steps))
         ;; MGL-backed transformer block
         (blk        (make-block d))
         (y          (forward blk x)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))))

;;; ------------------------------------------------------------
;;; stack of blocks
;;; ------------------------------------------------------------

(test stacked-blocks-forward-shape
  (let* ((d          32)
         (time-steps 10)
         (x          (random-mat d time-steps))
         (blocks     (loop repeat 3 collect (make-block d)))
         (stack      (make-block-list blocks))
         (y          (forward stack x)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))))

;;; ------------------------------------------------------------
;;; layer norm
;;; ------------------------------------------------------------

(test layer-norm-shape
  (let* ((rows 4)
         (cols 8)
         (x   (random-mat rows cols))
         ;; gamma/beta NIL: backend uses its default parameters
         (y   (tb-layer-norm x nil nil)))
    ;; shape invariant
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))
    ;; semantic check: per-feature mean ≈ 0, variance ≈ 1
    (destructuring-bind (r c) (tb-tensor-shape y)
      (dotimes (j c)
        (let ((sum 0d0)
              (sqsum 0d0))
          (dotimes (i r)
            (let ((v (mref y i j)))
              (incf sum v)
              (incf sqsum (* v v))))
          (let* ((mean (/ sum r))
                 (var  (- (/ sqsum r) (* mean mean))))
            (is (< (abs mean) 1d-6))
            (is (< (abs (- var 1d0)) 1d-3))))))))

;;; ------------------------------------------------------------
;;; dropout
;;; ------------------------------------------------------------

(test dropout-shape-and-mode
  (let* ((rows 32)
         (cols 16)
         (p    0.5d0)
         (x       (random-mat rows cols))
         (y-train (tb-dropout x p :training-p t))
         (y-test  (tb-dropout x p :training-p nil)))
    ;; shapes stay the same
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y-train)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y-test)))

    ;; inference mode: should be (numerically) identical to x
    (let ((diff-sum 0d0))
      (destructuring-bind (r c) (tb-tensor-shape x)
        (dotimes (i r)
          (dotimes (j c)
            (incf diff-sum
                  (abs (- (mref y-test i j)
                          (mref x i j)))))))
      (is (< diff-sum 1d-12)))

    ;; training mode: we expect roughly a fraction P of entries to be zeroed
    (let ((zeros 0)
          (total 0))
      (destructuring-bind (r c) (tb-tensor-shape y-train)
        (dotimes (i r)
          (dotimes (j c)
            (incf total)
            (when (zerop (mref y-train i j))
              (incf zeros))))
        (let* ((frac (if (plusp total)
                         (/ zeros total)
                         0d0)))
          ;; Very loose tolerance to avoid flakes; we just want to know
          ;; dropout is doing *something* like the right probability.
          (is (< (abs (- frac p)) 0.25d0)))))))
