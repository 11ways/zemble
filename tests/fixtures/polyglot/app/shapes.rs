pub trait Shape {
    fn area(&self) -> f64;
}

pub struct Circle {
    pub radius: f64,
}

impl Shape for Circle {
    fn area(&self) -> f64 {
        scale(3.0 * self.radius * self.radius)
    }
}

pub fn scale(value: f64) -> f64 {
    value * 2.0
}
