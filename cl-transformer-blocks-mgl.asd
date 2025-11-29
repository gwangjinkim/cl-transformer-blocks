(asdf:defsystem "cl-transformer-blocks-mgl"
  :description "MGL-MAT backend for cl-transformer-blocks-core (CPU/GPU capable)."
  :author "Gwang-Jin Kim <your@email>"
  :license "MIT"
  :depends-on ("cl-transformer-blocks-core" "mgl-mat")
  :serial t
  :pathname "src/mgl/"
  :components ((:file "package")
               (:file "backend")
               (:file "ops")
               (:file "factory")))
