library(stats)
area <- function(x, y = 1) {
  helper(x) + y
}
helper = function(n) n + 1
setClass("Point", representation(x = "numeric"))
