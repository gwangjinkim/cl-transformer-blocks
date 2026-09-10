(defpackage #:cl-transformer-blocks
  (:use #:cl)
  (:nicknames #:tb)
  (:export #:make-backend #:backend #:backend-device #:backend-memory
           #:reset-backend-peak-memory #:tensor #:tensor-shape #:tensor-dtype
           #:tensor-array #:tensor-from-array #:dispose #:with-resource
           #:from-pretrained #:download-from-hub #:save-pretrained #:model #:model-config
           #:make-portable-config #:make-portable-causal-lm #:python-architecture-files
           #:rotary-attention #:swiglu #:transformer-block #:composed-causal-lm
           #:make-rotary-attention #:make-swiglu #:make-transformer-block #:make-transformer
           #:component-config #:transformer-blocks #:save-composed-pretrained
           #:save-recomposed-pretrained
           #:model-backend #:named-parameters #:forward #:make-cache
           #:cache-length #:cache-capacity
           #:inspect-pretrained #:model-capabilities
           #:encode-text #:encode-batch #:decode-tokens #:attach-tokenizer #:generate #:sgd-step
           #:python-model #:python-worker-alive-p #:python-worker-info
           #:python-forward #:python-process #:python-processor-forward
           #:python-generate #:python-processor-generate
           #:make-chat-message #:python-apply-chat-template
           #:python-train-step #:python-processor-train-step #:make-python-file-input
           #:python-train-microbatches #:configure-python-scheduler #:python-optimizer-info
           #:python-adapter-info #:python-make-lora #:python-load-adapter
           #:python-save-adapter #:python-merge-adapter
           #:save-training-checkpoint #:restore-training-checkpoint #:push-to-hub
           #:load-adapter #:save-adapter #:make-lora #:merge-adapter #:remove-adapter
           #:trainable-parameters #:loss-and-gradients #:train-step #:make-sgd #:make-adamw
           #:distillation-loss-and-gradients #:distill-step
           #:distillation-batch #:make-distillation-batch #:make-top-k-distillation-batch
           #:save-distillation-batch
           #:load-distillation-batch #:distillation-batch-loss-and-gradients
           #:distillation-batch-storage-dtype #:distillation-batch-representation
           #:distillation-batch-top-k #:distillation-batch-temperature
           #:distill-batch-step
           #:distillation-example #:make-distillation-example
           #:distillation-example-input-ids #:distillation-example-labels
           #:distillation-example-attention-mask #:map-distillation-examples
           #:distillation-dataset #:save-distillation-dataset #:load-distillation-dataset
           #:save-distillation-dataset-from-teacher
           #:distillation-dataset-id #:distillation-dataset-size
           #:distillation-dataset-content-sha256
           #:distillation-dataset-epoch #:distillation-dataset-position
           #:start-distillation-dataset-epoch #:load-distillation-dataset-batch
           #:next-distillation-dataset-batch #:distill-dataset-step
           #:save-distillation-dataset-state #:restore-distillation-dataset-state
           #:compatibility-error #:backend-error #:shape-error))
