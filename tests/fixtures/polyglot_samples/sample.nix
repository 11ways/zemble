{ pkgs, lib }:
let
  helper = n: n + 1;
  area = x: helper x;
in {
  inherit area;
  point = { x = 1; };
}
