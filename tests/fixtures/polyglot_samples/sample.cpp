#include <vector>
namespace ns {
class Base { public: virtual int area() const = 0; };
struct Point : public Base, private Other { int x; int area() const override { return helper(x); } static int count; };
template <typename T> class Box { T value; public: Box(T v) : value(v) {} T get() const { return value; } };
int helper(int n) { return n; }
void run() { Point p; p.area(); ns::helper(1); auto b = new Box<int>(1); }
}
enum class Color { Red, Green };
