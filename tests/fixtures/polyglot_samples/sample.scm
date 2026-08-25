(define (area x) (helper x))
(define (helper n) (+ n 1))
(define limit 3)
(define-record-type point (make-point x) point? (x point-x))
