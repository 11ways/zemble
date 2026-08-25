import Foundation
protocol Shape { func area() -> Double }
class Base {}
class Point: Base, Shape {
  var x: Int = 0
  init(x: Int) { self.x = x }
  func area() -> Double { return helper(x) }
  static func make() -> Point { return Point(x: 1) }
}
struct Vec { var x: Int; func len() -> Int { return x } }
enum Color { case red, green }
extension Point { func extra() { self.area() } }
func helper(_ n: Int) -> Double { Double(n) }
