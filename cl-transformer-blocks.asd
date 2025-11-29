(asdf:defsystem "cl-transformer-blocks"
  :description "TransformerBlocks-style core + MGL backend for Common Lisp"
  :author "Gwang-Jin Kim <gwang.jin.kim.phd@gmail.com>"
  :license "MIT"
  :depends-on ("mgl-mat")
  :components
  ((:module "src"
    :components
    ((:module "core"
      :components
      ((:file "package")
       (:file "protocol")
       (:file "block")))
     (:module "mgl"
      :components
      ((:file "package")
       (:file "backend")
       (:file "factory")
       (:file "ops")))))))

(asdf:defsystem "cl-transformer-blocks/tests"
  :description "Test suite for cl-transformer-blocks"
  :depends-on ("cl-transformer-blocks" "fiveam")
  :components
  ((:module "tests"
    :components
    ((:file "package")
     (:file "mgl-tests" :depends-on ("package")))))
  :perform (asdf:test-op (op system)
             (declare (ignore op system))
             (uiop:symbol-call :cl-transformer-blocks.tests '#:run-tests)))
