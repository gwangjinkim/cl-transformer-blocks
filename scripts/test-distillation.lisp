(load "scripts/load.lisp")
(asdf:load-system "cl-transformer-blocks/tests")
(load "tests/distillation.lisp")
(tb-tests::run-distillation-tests)
