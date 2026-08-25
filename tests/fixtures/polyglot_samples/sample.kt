interface Shape { fun area(): Double }
class P(val x: Int) : Base(), Shape {
  private var count = 0
  override fun area(): Double = helper(x)
}
