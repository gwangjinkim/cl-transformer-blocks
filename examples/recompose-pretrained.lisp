;; Builds the full pretrained component checkpoint and prints its original output.
(load "examples/pretrained-components.lisp")

(let ((device (if (equal (uiop:getenv "TB_DEVICE") "gpu") :gpu :cpu))
      (destination ".build/recomposed-example/"))
  (tb:with-resource (source (tb:from-pretrained ".build/pretrained-components/" :device device))
    (let* ((count (length (tb:transformer-blocks source)))
           (indices (loop for index below count by 2 collect index)))
      (tb:save-recomposed-pretrained source destination indices :max-shard-size 16777216)
      (format t "~&Selected ~D of ~D source blocks: ~S~%" (length indices) count indices)))
  (tb:with-resource (edited (tb:from-pretrained destination :device device))
    (format t "Edited stack: ~D blocks, ~D canonical parameter tensors.~%"
            (length (tb:transformer-blocks edited)) (length (tb:named-parameters edited)))
    (let ((tokens (tb:encode-text edited "Common Lisp is a programming language" :add-special-tokens nil)))
      (format t "Edited model output: ~A~%"
              (tb:decode-tokens edited (tb:generate edited tokens :max-new-tokens 16) :skip-special-tokens t)))
    (format t "~&This changes the model's function; useful quality requires evaluation and potentially retraining.~%")))
