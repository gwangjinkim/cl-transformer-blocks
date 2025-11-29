;;; Backend-specific constructors that actually create wheight matrices and wire up block + sublayers.

;;; src/mgl/block.lisp

(in-package #:cl-transformer-blocks-mgl)

;;; ------------------------------------------------------------
;;; FFN configuration
;;; ------------------------------------------------------------

(defparameter *default-ffn-multiplier* 4
  "Multiplier for the FFN hidden dimension: H = *DEFAULT-FFN-MULTIPLIER* * D.")

;;; ------------------------------------------------------------
;;; Backend-specific constructors that actually create weight
;;; matrices and wire up block + sublayers.
;;; ------------------------------------------------------------

(defun make-attention-layer (model-dim)
  "Create an ATTENTION-LAYER with random initialized weights for MODEL-DIM."
  (make-instance 'cl-transformer-blocks:attention-layer
                 :w-q (random-mat model-dim model-dim)
                 :w-k (random-mat model-dim model-dim)
                 :w-v (random-mat model-dim model-dim)
                 :w-o (random-mat model-dim model-dim)
                 :model-dim model-dim))

(defun make-feedforward-layer (model-dim &key (multiplier *default-ffn-multiplier*))
  "Create a FEEDFORWARD-LAYER with hidden dimension = MULTIPLIER * MODEL-DIM."
  (let ((hidden-dim (* multiplier model-dim)))
    (make-instance 'cl-transformer-blocks:feedforward-layer
                   :w1 (random-mat hidden-dim model-dim)
                   :w2 (random-mat model-dim hidden-dim)
                   :model-dim model-dim
                   :hidden-dim hidden-dim)))

(defun make-block (model-dim &key (ffn-multiplier *default-ffn-multiplier*)
                              (use-layer-norm t))
  "Create a full TRANSFORMER-BLOCK (attention + FFN) with random parameters."
  (let* ((attn (make-attention-layer model-dim))
         (ffn  (make-feedforward-layer model-dim :multiplier ffn-multiplier)))
    (make-instance 'cl-transformer-blocks:transformer-block
                   :attention attn
                   :ffn       ffn
                   :use-layer-norm use-layer-norm)))
