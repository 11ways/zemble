local M = {}
local count = 0
function M.area(x) return helper(x) end
function M:run(n) self.area(n); M.area(1) end
local function helper(n) return n end
function top(n) return top(n - 1) end
return M
