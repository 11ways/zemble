pragma solidity ^0.8.0;
import "./Base.sol";
interface IShape { function area() external view returns (uint); }
contract Point is Base, IShape {
  uint x;
  event Moved(uint x);
  modifier onlyOwner() { _; }
  constructor(uint _x) { x = _x; }
  function area() external view override returns (uint) { return helper(x); }
  function helper(uint n) internal pure returns (uint) { return n; }
}
library Helper { function compute(uint n) internal pure returns (uint) { return n; } }
struct Vec { uint x; }
enum Color { Red, Green }
