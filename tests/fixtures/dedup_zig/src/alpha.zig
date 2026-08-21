const std = @import("std");

pub fn sumAll(items: []const u32) u32 {
    var total: u32 = 0;
    var index: usize = 0;
    while (index < items.len) : (index += 1) {
        if (items[index] > 10) {
            total += items[index];
        }
    }
    return total;
}

fn scaleValues(values: []u32, factor: u32) u32 {
    var carry: u32 = 0;
    var cursor: usize = 0;
    while (cursor < values.len) : (cursor += 1) {
        values[cursor] = values[cursor] * factor;
        carry += values[cursor];
    }
    return carry;
}

const Widget = struct {
    label: u32,

    pub fn fill(alloc: std.mem.Allocator, count: usize) ![]u8 {
        const buffer = try alloc.alloc(u8, count);
        var index: usize = 0;
        while (index < count) : (index += 1) {
            buffer[index] = 0;
        }
        std.debug.print("filled {d}\n", .{count});
        return buffer;
    }
};
