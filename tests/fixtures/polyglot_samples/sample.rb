require 'json'
module Store
  class Point < Base
    include Comparable
    attr_reader :x
    def initialize(x)
      @x = x
    end
    def self.build(a)
      new(a)
    end
    def area(n = 1)
      helper(n)
      @x.compute(1, 2)
    end
    private
    def helper(n); n; end
  end
end
def top(n)
  top(n - 1)
end
