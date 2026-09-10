;;;; A bounded application language and deterministic teacher, entirely in Lisp.
(defpackage #:tb-query-teacher (:use #:cl))
(in-package #:tb-query-teacher)

(defstruct query status component period)
(defstruct sample request query split held-combination-p)

(defun all-queries ()
  (loop for status in '(:open :closed :any) append
    (loop for component in '(:tokenizer :training :storage :any) append
      (loop for period in '(:week :month :any)
            collect (make-query :status status :component component :period period)))))

(defun render-query (query)
  (format nil "(issues :~(~A~) :~(~A~) :~(~A~))"
          (query-status query) (query-component query) (query-period query)))

(defun parse-query (text)
  "Accept only the 36 canonical commands; never invoke the Lisp reader or EVAL."
  (when (and (stringp text) (<= (length text) 128))
    (let ((trimmed (string-trim '(#\Space #\Tab #\Newline #\Return) text)))
      (find trimmed (all-queries) :test #'string= :key #'render-query))))

(defun example-issues ()
  ;; IDs, status, component, integer age in days; independent of wall-clock date.
  '((1 :open :tokenizer 2) (2 :closed :tokenizer 8)
    (3 :open :training 6) (4 :closed :storage 29)
    (5 :open :storage 30) (6 :closed :training 40)))

(defun execute-query (query issues)
  "Return matching IDs. :WEEK means age < 7; :MONTH means age < 30."
  (unless (and (typep query 'query) (parse-query (render-query query)))
    (error "Invalid issue query"))
  (loop for (id status component age) in issues
        when (and (or (eq :any (query-status query)) (eq status (query-status query)))
                  (or (eq :any (query-component query)) (eq component (query-component query)))
                  (case (query-period query)
                    (:week (< age 7)) (:month (< age 30)) (:any t)))
          collect id))

(defparameter *test-requests*
  ;; Separately authored evaluation sentences, one per canonical command.
  ;; This is a small synthetic benchmark, not a representative user study.
  #("Which tokenizer problems remain open from the last seven days?"
    "Find unresolved tokenization reports younger than thirty days."
    "I need all open tokenizer tickets, however old."
    "What training bugs are still open and less than a week old?"
    "List unresolved training issues from the past thirty days."
    "Find every open training report, without a date restriction."
    "Show unresolved storage problems reported during the last week."
    "Which storage tickets from the last month are still open?"
    "Give me the open storage backlog across all dates."
    "Across all components, find open reports from the last seven days."
    "I want unresolved tickets from the last thirty days, in any component."
    "Show the entire open backlog, regardless of component or age."
    "Which closed tokenizer reports are less than seven days old?"
    "Find tokenizer issues that are closed and from the past month."
    "Retrieve all closed tokenizer reports, including older ones."
    "Show closed training tickets reported less than a week ago."
    "Find closed training problems from the last thirty days."
    "I want every closed training issue, from any date."
    "Which storage reports from this past week are closed?"
    "List closed storage tickets younger than thirty days."
    "Show closed storage reports with no age limit."
    "Find closed tickets from the last week across every component."
    "Which reports are closed and under thirty days old, regardless of component?"
    "Retrieve the complete closed backlog across all components and dates."
    "Show tokenizer reports from the last week, both open and closed."
    "Find all tokenizer tickets younger than thirty days, whatever their status."
    "Give me every tokenizer issue, without status or date filters."
    "List training reports from the past seven days, including closed ones."
    "Which training issues were reported in the past month, whether open or closed?"
    "Show the full training issue history, in either status."
    "Find storage tickets from the last seven days in any status."
    "Show all storage reports under thirty days old, open or closed."
    "Retrieve the complete storage backlog, without status or age restrictions."
    "Show every issue reported during the last week, across components and statuses."
    "Find all reports from the past thirty days, regardless of component or status."
    "List every issue. Do not filter by status, component, or date."))

(defun training-request (query variant)
  (let ((status (ecase (query-status query)
                  (:open "open") (:closed "closed") (:any "open and closed")))
        (component (ecase (query-component query)
                     (:tokenizer "tokenizer") (:training "training")
                     (:storage "storage") (:any "all components")))
        (period (ecase (query-period query)
                  (:week "the last seven days") (:month "the last thirty days")
                  (:any "all dates"))))
    (format nil (elt '("Show ~A issues for ~A from ~A."
                       "Find ~A tickets in ~A covering ~A."
                       "I need ~A reports about ~A from ~A."
                       "Search ~A bugs for ~A over ~A."
                       "Retrieve ~A issues: ~A; ~A."
                       "Please list ~A tickets concerning ~A, for ~A."
                       "Could you fetch ~A reports for ~A spanning ~A?") variant)
            status component period)))

(defun make-curriculum ()
  (loop for query in (all-queries) for index from 0
        for held = (zerop (mod index 7)) append
    (append
     (unless held
       (loop for variant below 7 collect
         (make-sample :request (training-request query variant) :query query
                      :split (if (= variant 6) :validation :train))))
     (list (make-sample :request (aref *test-requests* index) :query query
                        :split :test :held-combination-p held)))))

(defun prompt-for (request)
  (format nil "Translate the request into one issue query.~%Syntax: (issues STATUS COMPONENT PERIOD)~%STATUS: :open, :closed, :any. COMPONENT: :tokenizer, :training, :storage, :any. PERIOD: :week, :month, :any.~%Request: ~A~%Query:" request))
