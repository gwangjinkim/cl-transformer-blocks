(asdf:defsystem "cl-transformer-blocks"
  :description "Native MLX Transformer models with Hugging Face interchange"
  :version "0.38.0"
  :author "Gwang-Jin Kim"
  :license "MIT"
  :depends-on ("alexandria" "cffi" "yason" "trivial-garbage")
  :serial t
  :components ((:file "src/package") (:file "src/protocol") (:file "src/sha256")
               (:file "src/mlx") (:file "src/huggingface")
               (:file "src/llama") (:file "src/portable") (:file "src/components")
               (:file "src/composition-import")
               (:file "src/qwen2") (:file "src/gpt2") (:file "src/bert")
               (:file "src/generation")
               (:file "src/python-worker")
               (:file "src/adapters") (:file "src/training") (:file "src/optimizers")
               (:file "src/distillation")
               (:module "python"
                :components
                ((:module "tb_parallel"
                  :components ((:static-file "configuration_tb_parallel.py")
                               (:static-file "modeling_tb_parallel.py")))
                 (:module "tb_composed"
                  :components ((:static-file "configuration_tb_composed.py")
                               (:static-file "modeling_tb_composed.py"))))))
  :in-order-to ((asdf:test-op (asdf:test-op "cl-transformer-blocks/tests"))))
(asdf:defsystem "cl-transformer-blocks/tests"
  :depends-on ("cl-transformer-blocks")
  :serial t
  :components ((:file "tests/suite"))
  :perform (asdf:test-op (op system)
             (declare (ignore op system))
             (uiop:symbol-call :tb-tests :run-tests)))
