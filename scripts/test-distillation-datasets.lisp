(load "scripts/load.lisp")
(asdf:load-system "cl-transformer-blocks/tests")
(load "tests/distillation-datasets.lisp")
(tb-tests::run-distillation-dataset-tests)
