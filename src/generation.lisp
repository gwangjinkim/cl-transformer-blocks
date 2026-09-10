(in-package #:tb)
(defclass native-tokenizer (native-resource) ())
(defvar *tokenizer-loaded* nil)
(defun ensure-tokenizer-library ()
  (unless *tokenizer-loaded*
    (cffi:load-foreign-library
     (or (uiop:getenv "TB_TOKENIZER_LIBRARY")
         (namestring (merge-pathnames
                      #+darwin ".build/tokenizer/release/libtb_tokenizer.dylib"
                      #-darwin ".build/tokenizer/release/libtb_tokenizer.so"
                      (asdf:system-source-directory "cl-transformer-blocks")))))
    (setf *tokenizer-loaded* t)))
(define-native "tb_tokenizer_load" :pointer (path :string))
(define-native "tb_tokenizer_from_bytes" :pointer (data :pointer) (length :size)
               (vocabulary-size :uint32))
(define-native "tb_tokenizer_free" :void (tokenizer :pointer))
(define-native "tb_tokenizer_error" :string)
(define-native "tb_tokenizer_encode" :pointer (tokenizer :pointer) (text :string) (special :int))
(define-native "tb_tokenizer_encode_batch" :pointer (tokenizer :pointer) (request :string))
(define-native "tb_tokenizer_decode" :pointer (tokenizer :pointer) (ids :string) (skip :int))
(define-native "tb_tokenizer_string_free" :void (string :pointer))
(define-native "tb_next_token" :int (context :pointer) (logits :pointer) (token :pointer))
(defun tokenizer-pointer (p)
  (when (cffi:null-pointer-p p) (fail 'compatibility-error "Tokenizer: ~A" (%tb-tokenizer-error))) p)
(defun ensure-tokenizer (model)
  (pointer (model-backend model))
  (or (model-tokenizer model)
      (progn
        (unless (and (model-directory model)
                     (probe-file (merge-pathnames "tokenizer.json" (model-directory model))))
          (fail 'compatibility-error "Model has no tokenizer; use ATTACH-TOKENIZER first"))
        (ensure-tokenizer-library)
        (let* ((p (tokenizer-pointer (%tb-tokenizer-load
                   (namestring (merge-pathnames "tokenizer.json" (model-directory model))))))
               (tokenizer (make-instance 'native-tokenizer :handle p)))
          (trivial-garbage:finalize tokenizer (lambda () (%tb-tokenizer-free p)))
          (setf (model-tokenizer model) tokenizer)))))
(defparameter +tokenizer-asset-names+
  '("tokenizer.json" "tokenizer_config.json" "special_tokens_map.json"
    "added_tokens.json" "tokenizer.model" "vocab.json" "merges.txt"
    "vocab.txt" "chat_template.jinja"))

(defmethod attach-tokenizer ((model model) directory)
  "Attach a snapshot; vocabulary IDs must fit the existing embedding table.
This checks ID bounds, not the semantic agreement of token names and trained weights."
  (pointer (model-backend model))
  (let* ((directory (uiop:ensure-directory-pathname directory))
         (assets
           (loop for name in +tokenizer-asset-names+
                 for path = (merge-pathnames name directory)
                 when (probe-file path)
                   collect (cons name
                                 (with-open-file (stream path :element-type '(unsigned-byte 8))
                                   (let ((bytes (make-array (file-length stream)
                                                            :element-type '(unsigned-byte 8))))
                                     (unless (= (read-sequence bytes stream) (length bytes))
                                       (fail 'compatibility-error "Incomplete tokenizer asset: ~A" name))
                                     bytes)))))
         (data (cdr (assoc "tokenizer.json" assets :test #'equal))))
    (unless (and data (plusp (length data)))
      (fail 'compatibility-error "Tokenizer directory must contain a nonempty tokenizer.json"))
    (ensure-tokenizer-library)
    ;; Rust parses synchronously and retains no pointer into the pinned Lisp vector.
    (cffi:with-pointer-to-vector-data (buffer data)
      (let* ((p (tokenizer-pointer
                 (%tb-tokenizer-from-bytes buffer (length data) (config model "vocab_size"))))
             (tokenizer (make-instance 'native-tokenizer :handle p)))
        (trivial-garbage:finalize tokenizer (lambda () (%tb-tokenizer-free p)))
        (dispose (model-tokenizer model))
        (setf (model-tokenizer model) tokenizer
              (model-tokenizer-assets model) assets))))
  model)
(defmethod dispose ((tokenizer native-tokenizer))
  (unless (cffi:null-pointer-p (handle tokenizer))
    (trivial-garbage:cancel-finalization tokenizer)
    (%tb-tokenizer-free (handle tokenizer)) (setf (handle tokenizer) (cffi:null-pointer))))
(defun consume-tokenizer-string (p)
  (tokenizer-pointer p)
  (unwind-protect (cffi:foreign-string-to-lisp p :encoding :utf-8)
    (%tb-tokenizer-string-free p)))
(defmethod encode-text ((model model) text &key (add-special-tokens t))
  "Run tokenizer.json through native Hugging Face Tokenizers; return an ID vector."
  (unless (and (stringp text) (not (find #\Null text)))
    (fail 'compatibility-error "Text must be a string without embedded NUL characters"))
  (let ((ids (yason:parse (consume-tokenizer-string
                (%tb-tokenizer-encode (pointer (ensure-tokenizer model)) text (if add-special-tokens 1 0)))
               :json-arrays-as-vectors t)))
    (unless (every (lambda (id) (typep id `(integer 0 (,(config model "vocab_size"))))) ids)
      (fail 'compatibility-error "Tokenizer emitted IDs outside the model vocabulary"))
    ids))
(defun tokenizer-text-sequence (texts)
  (unless (and (or (and (listp texts) (alexandria:proper-list-p texts))
                   (and (vectorp texts) (not (stringp texts))))
               (plusp (length texts))
               (every (lambda (text) (and (stringp text) (not (find #\Null text)))) texts))
    (fail 'compatibility-error "Expected a nonempty sequence of strings without embedded NUL characters"))
  (coerce texts 'vector))

(defun tokenizer-batch-array (response key rows columns upper-bound)
  (let ((values (gethash key response))
        (result (make-array (list rows columns) :element-type '(unsigned-byte 32))))
    (unless (and (vectorp values) (= (length values) rows))
      (fail 'compatibility-error "Tokenizer returned invalid batch rows: ~A" key))
    (dotimes (row rows result)
      (let ((values (aref values row)))
        (unless (and (vectorp values) (= (length values) columns))
          (fail 'compatibility-error "Tokenizer returned ragged batch rows: ~A" key))
        (dotimes (column columns)
          (let ((value (aref values column)))
            (unless (typep value `(integer 0 (,upper-bound)))
              (fail 'compatibility-error "Tokenizer emitted out-of-range ~A" key))
            (setf (aref result row column) value)))))))

(defmethod encode-batch ((model model) texts &key text-pairs (add-special-tokens t)
                         (padding :longest) max-length (truncation nil)
                         (pad-token-id (config model "pad_token_id")))
  "Encode single texts or pairs to three owned Lisp arrays; no implicit truncation.
PADDING is :LONGEST or :MAX-LENGTH. TRUNCATION T means right longest-first.
MAX-LENGTH is a ceiling including special tokens and cannot exceed model context."
  (pointer (model-backend model))
  (let* ((texts (tokenizer-text-sequence texts))
         (pairs (when text-pairs (tokenizer-text-sequence text-pairs)))
         (context (model-context-length model)))
    (unless (and (or (null pairs) (= (length texts) (length pairs)))
                 (member padding '(:longest :max-length))
                 (member truncation '(nil t))
                 (or max-length (and (not truncation) (eq padding :longest)))
                 (or (null max-length) (typep max-length `(integer 1 ,context)))
                 (or (null pad-token-id)
                     (typep pad-token-id `(integer 0 (,(config model "vocab_size"))))))
      (fail 'compatibility-error "Invalid batch pairs, padding, truncation, length, or padding token ID"))
    (let* ((request (make-hash-table :test 'equal)))
      (setf (gethash "texts" request) texts
            (gethash "text_pairs" request) pairs
            (gethash "add_special_tokens" request) (if add-special-tokens 'yason:true 'yason:false)
            (gethash "padding" request) (string-downcase padding)
            (gethash "max_length" request) (or max-length context)
            (gethash "truncation" request) (if truncation 'yason:true 'yason:false)
            (gethash "pad_token_id" request) pad-token-id)
      (let* ((response (yason:parse
                        (consume-tokenizer-string
                         (%tb-tokenizer-encode-batch
                          (pointer (ensure-tokenizer model))
                          (with-output-to-string (stream) (yason:encode request stream))))
                        :json-arrays-as-vectors t))
             (rows (length texts))
             (columns (length (aref (gethash "input_ids" response) 0))))
        (unless (<= 1 columns (or max-length context))
          (fail 'compatibility-error "Tokenizer returned invalid batch width"))
        (values (tokenizer-batch-array response "input_ids" rows columns (config model "vocab_size"))
                (tokenizer-batch-array response "attention_mask" rows columns 2)
                (tokenizer-batch-array response "token_type_ids" rows columns (expt 2 32)))))))

(defmethod decode-tokens ((model model) ids &key (skip-special-tokens nil))
  "Decode IDs using the checkpoint's native tokenizer (no Python cleanup wrapper)."
  (unless (and (vectorp ids) (every (lambda (id) (typep id '(unsigned-byte 32))) ids))
    (fail 'shape-error "Expected a vector of nonnegative token IDs"))
  (consume-tokenizer-string
   (%tb-tokenizer-decode (pointer (ensure-tokenizer model))
                         (with-output-to-string (s) (yason:encode ids s)) (if skip-special-tokens 1 0))))
(defun next-token (backend logits)
  (cffi:with-foreign-object (token :int)
    (checked-status (%tb-next-token (pointer backend) (pointer logits) token))
    (cffi:mem-ref token :int)))
(defmethod generate ((model model) prompt &key (max-new-tokens 20)
                                      (eos-token-id (config model "eos_token_id")))
  "Greedy, single-sequence cached decoding; return prompt plus generated IDs.
PROMPT is an ID vector or text. EOS may be an integer, a sequence, or NIL to disable."
  (unless (typep max-new-tokens '(integer 0)) (fail 'shape-error "Invalid token limit"))
  (let* ((ids (if (stringp prompt) (encode-text model prompt) prompt))
         (out (make-array 0 :adjustable t :fill-pointer 0))
         (eos (cond ((null eos-token-id) nil) ((integerp eos-token-id) (list eos-token-id))
                    (t (coerce eos-token-id 'list)))))
    (unless (and (vectorp ids) (> (length ids) 0)) (fail 'shape-error "Prompt must be a nonempty token vector"))
    (loop for id across ids do (vector-push-extend id out))
    (with-resource (cache (make-cache model))
      (let ((input (make-array (list 1 (length ids)) :initial-contents (list ids))))
        (dotimes (i max-new-tokens)
          (with-resource (logits (forward model input :cache cache))
            (let ((token (next-token (model-backend model) logits)))
              (vector-push-extend token out)
              (when (member token eos) (return))
              (setf input (make-array '(1 1) :initial-element token))))))) out))
