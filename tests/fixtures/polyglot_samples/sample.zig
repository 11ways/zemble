const std = @import("std");
pub const Point = struct {
    x: i32,
    pub fn area(self: Point) i32 { return helper(self.x); }
    pub fn init(x: i32) Point { return .{ .x = x }; }
};
const Color = enum { red, green };
fn helper(n: i32) i32 { return n + 1; }
pub fn run() void { const p = Point.init(1); _ = p.area(); std.debug.print("x", .{}); }
test "area" { try std.testing.expect(helper(1) == 2); }
