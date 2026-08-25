import 'dart:math';
abstract class Shape { double area(); }
class Base {}
class Point extends Base implements Shape {
  int x = 0;
  Point(this.x);
  @override
  double area() => helper(x);
  static Point make() => Point(1);
  void run() { this.area(); Helper.compute(1); }
}
mixin Greets { void hi() {} }
enum Color { red, green }
double helper(int n) => n.toDouble();
