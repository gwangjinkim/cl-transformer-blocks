;;; Composes multiple blocks sequentially.

(in-package #:cl-transformer-blocks)

(defclass block-list ()
  ((blocks :initarg :blocks :reader block-list-blocks))
  (:documentation "Sequential list of modules with a FORWARD method (e.g., BLOCKs)."))

(defun make-block-list (blocks)
  "Wrap a list or vector of module instances into a BLOCK-LIST."
  (make-instance 'block-list :blocks (coerce blocks 'vector)))

(defmethod forward ((bl block-list) x &key mask training-p)
  (let ((result x))
    (loop for blk across (block-list-blocks bl) do
         (setf result (forward blk result :mask mask :training-p training-p)))
    result))
