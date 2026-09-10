(load "scripts/load.lisp")
(asdf:load-system "cl-transformer-blocks/tests")
(load "tests/distillation-targets.lisp")
(tb-tests::run-distillation-target-tests)
