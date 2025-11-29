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
  (let* ((rows 2)
         (cols 4)
         (x     (random-mat rows cols))
         ;; no gamma/beta: identity scale/shift
         (y     (tb-layer-norm x nil nil)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y)))))

;;; ------------------------------------------------------------
;;; dropout
;;; ------------------------------------------------------------

(test dropout-shape-and-mode
  (let* ((rows 2)
         (cols 4)
         (x       (random-mat rows cols))
         (y-train (tb-dropout x 0.5 :training-p t))
         (y-test  (tb-dropout x 0.5 :training-p nil)))
    ;; shapes stay the same
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y-train)))
    (is (equal (tb-tensor-shape x)
               (tb-tensor-shape y-test)))
    ;; in inference mode we expect no change (or at least very small diff)
    (let ((diff-sum 0d0))
      (destructuring-bind (r c) (tb-tensor-shape x)
        (dotimes (i r)
          (dotimes (j c)
            (incf diff-sum
                  (abs (- (mref y-test i j)
                          (mref x i j)))))))
      (is (< diff-sum 1d-8)))))
