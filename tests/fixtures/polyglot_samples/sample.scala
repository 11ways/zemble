package app.core
import scala.collection.mutable
trait Shape { def area(): Double }
abstract class Base
class Point(val x: Int) extends Base with Shape {
  private var count = 0
  override def area(): Double = helper(x)
  def run(n: Int): Unit = { this.area(); Helper.compute(n) }
}
object Helper { def compute(n: Int): Double = n.toDouble }
case class Vec(x: Int)
def helper(n: Int): Double = n.toDouble
