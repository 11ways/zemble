#lang racket
(require racket/list)
(define (area x) (helper x))
(define (helper n) (+ n 1))
(struct point (x y))
(define limit 3)
