(load "scripts/load.lisp")
(asdf:load-system "cl-transformer-blocks/tests")
(load "tests/components.lisp")
(tb-tests::run-component-roundtrip-tests)
