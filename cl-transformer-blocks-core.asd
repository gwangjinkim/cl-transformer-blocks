(asdf:defsystem "cl-transformer-blocks-core"
  :description "Core transformer blocks independent of numeric backend."
  :author "Gwang-Jin Kim <gwang.jin.kim.phd@gmail.com>"
  :license "MIT"
  :serial t
  :pathname "src/core/"
  :components ((:file "package")
               (:file "protocol")
               (:file "attention")
               (:file "feedforward")
               (:file "block")
               (:file "block-list")))
