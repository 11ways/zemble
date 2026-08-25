use std::fmt;
pub mod inner { pub fn nested() {} }
pub struct Point { pub x: i32, y: i32 }
pub enum Shape { Circle(f64), Square { side: f64 } }
pub trait Area { fn area(&self) -> f64; fn name(&self) -> String { String::new() } }
impl Area for Point { fn area(&self) -> f64 { helper(self.x) } }
impl Point { pub fn new(x: i32) -> Self { Point { x, y: 0 } } fn go(&self) { self.area(); Point::new(1); } }
pub fn helper(n: i32) -> f64 { n as f64 }
const LIMIT: i32 = 3;
static COUNT: i32 = 0;
type Alias = Point;
