-module(point).
-export([area/1]).
-record(point, {x, y}).
area(P) -> helper(P#point.x).
helper(N) -> lists:sum([N]).
