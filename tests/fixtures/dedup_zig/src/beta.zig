const std = @import("std");

const Helper = struct {
    seed: u32,

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
};

fn scaleAmounts(amounts: []u32, weight: u32) u32 {
    var running: u32 = 0;
    var position: usize = 0;
    while (position < amounts.len) : (position += 1) {
        amounts[position] = amounts[position] * weight;
        running += amounts[position];
    }
    return running;
}

fn fill(alloc: std.mem.Allocator, count: usize) ![]u8 {
    const buffer = try alloc.alloc(u8, count);
    var index: usize = 0;
    while (index < count) : (index += 1) {
        buffer[index] = 0;
    }
    std.debug.warn("filled {d}\n", .{count});
    return buffer;
}
