package app.core
import groovy.transform.ToString
interface Shape { double area() }
abstract class Base {}
class Point extends Base implements Shape {
  int x = 0
  Point(int x) { this.x = x }
  double area() { return helper(x) }
  static Point make() { new Point(1) }
  def run() { this.area(); Helper.compute(1) }
}
enum Color { RED, GREEN }
def helper(n) { n }
