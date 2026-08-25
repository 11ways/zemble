module Store
using LinearAlgebra
struct Point
  x::Int
end
abstract type Shape end
function area(p::Point)
  helper(p.x)
end
helper(n) = n + 1
macro m(x) x end
end
