;; From the repository root, after the real-model acceptance test downloads its fixture:
;; sbcl --no-sysinit --no-userinit --script examples/llama.lisp
(load "scripts/load.lisp")
(tb:with-resource (model (tb:from-pretrained ".build/models/smollm2/" :device :gpu))
  (let* ((prompt (tb:encode-text model "Common Lisp is" :add-special-tokens nil))
         (tokens (tb:generate model prompt :max-new-tokens 20)))
    (format t "~A~%" (tb:decode-tokens model tokens :skip-special-tokens t)))
  (tb:save-pretrained model ".build/example-export/"))
