(require :asdf)
(defparameter *project-root* (uiop:pathname-parent-directory-pathname
                              (uiop:pathname-directory-pathname *load-truename*)))
;; Structured translations retain source directory components: no package.fasl collisions.
(asdf:initialize-output-translations
 `(:output-translations (t (,(namestring (merge-pathnames ".cache/fasl/" *project-root*)) :implementation))
                        :ignore-inherited-configuration))
;; Reproducible test entry point: use only the project and locked local dependencies.
(asdf:initialize-source-registry
 `(:source-registry (:directory ,*project-root*)
                    (:tree ,(merge-pathnames ".build/deps/" *project-root*))
                    :ignore-inherited-configuration))
(asdf:load-system "cl-transformer-blocks")
