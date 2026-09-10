(in-package #:tb)

;; A small portable implementation keeps artifact verification available on every
;; supported Common Lisp without adding a platform crypto library dependency.
(defparameter *sha256-round-constants*
  (make-array
   64 :element-type '(unsigned-byte 32)
   :initial-contents
   '(#x428a2f98 #x71374491 #xb5c0fbcf #xe9b5dba5 #x3956c25b #x59f111f1 #x923f82a4 #xab1c5ed5
     #xd807aa98 #x12835b01 #x243185be #x550c7dc3 #x72be5d74 #x80deb1fe #x9bdc06a7 #xc19bf174
     #xe49b69c1 #xefbe4786 #x0fc19dc6 #x240ca1cc #x2de92c6f #x4a7484aa #x5cb0a9dc #x76f988da
     #x983e5152 #xa831c66d #xb00327c8 #xbf597fc7 #xc6e00bf3 #xd5a79147 #x06ca6351 #x14292967
     #x27b70a85 #x2e1b2138 #x4d2c6dfc #x53380d13 #x650a7354 #x766a0abb #x81c2c92e #x92722c85
     #xa2bfe8a1 #xa81a664b #xc24b8b70 #xc76c51a3 #xd192e819 #xd6990624 #xf40e3585 #x106aa070
     #x19a4c116 #x1e376c08 #x2748774c #x34b0bcb5 #x391c0cb3 #x4ed8aa4a #x5b9cca4f #x682e6ff3
     #x748f82ee #x78a5636f #x84c87814 #x8cc70208 #x90befffa #xa4506ceb #xbef9a3f7 #xc67178f2)))

(defstruct (sha256-context (:constructor make-sha256-context ()))
  (hash (make-array
         8 :element-type '(unsigned-byte 32)
         :initial-contents
         '(#x6a09e667 #xbb67ae85 #x3c6ef372 #xa54ff53a
           #x510e527f #x9b05688c #x1f83d9ab #x5be0cd19))
        :type (simple-array (unsigned-byte 32) (8)))
  (buffer (make-array 64 :element-type '(unsigned-byte 8))
          :type (simple-array (unsigned-byte 8) (64)))
  (buffer-length 0 :type (integer 0 64))
  (total-bytes 0 :type integer))

(declaim (inline sha256-u32 sha256-rotr))

(defun sha256-u32 (value)
  (logand value #xffffffff))

(defun sha256-rotr (value count)
  (sha256-u32 (logior (ash value (- count)) (ash value (- 32 count)))))

(defun sha256-process-block (context octets start)
  (let ((words (make-array 64 :element-type '(unsigned-byte 32)))
        (hash (sha256-context-hash context)))
    (dotimes (index 16)
      (let ((offset (+ start (* 4 index))))
        (setf (aref words index)
              (logior (ash (aref octets offset) 24)
                      (ash (aref octets (+ offset 1)) 16)
                      (ash (aref octets (+ offset 2)) 8)
                      (aref octets (+ offset 3))))))
    (loop for index from 16 below 64 do
      (let* ((x (aref words (- index 15)))
             (y (aref words (- index 2)))
             (sigma-zero (logxor (sha256-rotr x 7) (sha256-rotr x 18) (ash x -3)))
             (sigma-one (logxor (sha256-rotr y 17) (sha256-rotr y 19) (ash y -10))))
        (setf (aref words index)
              (sha256-u32 (+ (aref words (- index 16)) sigma-zero
                               (aref words (- index 7)) sigma-one)))))
    (let ((a (aref hash 0)) (b (aref hash 1)) (c (aref hash 2)) (d (aref hash 3))
          (e (aref hash 4)) (f (aref hash 5)) (g (aref hash 6)) (h (aref hash 7)))
      (dotimes (index 64)
        (let* ((sum-one (logxor (sha256-rotr e 6) (sha256-rotr e 11)
                                (sha256-rotr e 25)))
               (choose (logxor (logand e f) (logand (lognot e) g)))
               (temporary-one
                 (sha256-u32 (+ h sum-one choose (aref *sha256-round-constants* index)
                                  (aref words index))))
               (sum-zero (logxor (sha256-rotr a 2) (sha256-rotr a 13)
                                 (sha256-rotr a 22)))
               (majority (logxor (logand a b) (logand a c) (logand b c)))
               (temporary-two (sha256-u32 (+ sum-zero majority))))
          (setf h g g f f e e (sha256-u32 (+ d temporary-one))
                d c c b b a a (sha256-u32 (+ temporary-one temporary-two)))))
      (setf (aref hash 0) (sha256-u32 (+ (aref hash 0) a))
            (aref hash 1) (sha256-u32 (+ (aref hash 1) b))
            (aref hash 2) (sha256-u32 (+ (aref hash 2) c))
            (aref hash 3) (sha256-u32 (+ (aref hash 3) d))
            (aref hash 4) (sha256-u32 (+ (aref hash 4) e))
            (aref hash 5) (sha256-u32 (+ (aref hash 5) f))
            (aref hash 6) (sha256-u32 (+ (aref hash 6) g))
            (aref hash 7) (sha256-u32 (+ (aref hash 7) h)))))
  context)

(defun sha256-update (context octets &key (start 0) (end (length octets)))
  (unless (and (typep octets '(array (unsigned-byte 8) (*)))
               (typep start `(integer 0 ,(length octets)))
               (typep end `(integer ,start ,(length octets))))
    (error "SHA-256 input must be an octet vector with valid bounds"))
  (incf (sha256-context-total-bytes context) (- end start))
  (loop while (< start end) do
    (let* ((available (- 64 (sha256-context-buffer-length context)))
           (count (min available (- end start)))
           (buffer-start (sha256-context-buffer-length context)))
      (replace (sha256-context-buffer context) octets
               :start1 buffer-start :end1 (+ buffer-start count)
               :start2 start :end2 (+ start count))
      (incf (sha256-context-buffer-length context) count)
      (incf start count)
      (when (= (sha256-context-buffer-length context) 64)
        (sha256-process-block context (sha256-context-buffer context) 0)
        (setf (sha256-context-buffer-length context) 0))))
  context)

(defun sha256-finalize (context)
  (let* ((bit-length (mod (* 8 (sha256-context-total-bytes context)) (expt 2 64)))
         (fill (sha256-context-buffer-length context))
         (padding-length (if (< fill 56) (- 56 fill) (- 120 fill)))
         (padding (make-array (+ padding-length 8) :element-type '(unsigned-byte 8)
                              :initial-element 0)))
    (setf (aref padding 0) #x80)
    (dotimes (index 8)
      (setf (aref padding (+ padding-length index))
            (ldb (byte 8 (* 8 (- 7 index))) bit-length)))
    (sha256-update context padding)
    (string-downcase
     (format nil "~{~8,'0x~}" (coerce (sha256-context-hash context) 'list)))))

(defun sha256-octets (octets)
  (sha256-finalize (sha256-update (make-sha256-context) octets)))

(defun sha256-ascii-string (string)
  (let ((octets (make-array (length string) :element-type '(unsigned-byte 8))))
    (dotimes (index (length string))
      (let ((code (char-code (char string index))))
        (unless (< code 128)
          (error "SHA-256 canonical text must contain only ASCII characters"))
        (setf (aref octets index) code)))
    (sha256-octets octets)))

(defun sha256-file (path)
  (with-open-file (stream path :direction :input :element-type '(unsigned-byte 8))
    (let ((context (make-sha256-context))
          (buffer (make-array 65536 :element-type '(unsigned-byte 8))))
      (loop for count = (read-sequence buffer stream)
            while (plusp count) do (sha256-update context buffer :end count))
      (sha256-finalize context))))
