module Point = struct
  type t = { x : int; y : int }
  let area p = helper p.x
  let make x = { x; y = 0 }
end
let helper n = n + 1
type shape = Circle of float | Square of float
class point x = object method area = x end
