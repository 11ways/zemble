<?php
namespace App\Core;
use App\Base;
interface Shape { public function area(): float; }
abstract class Point extends Base implements Shape {
  private int $x = 0;
  public const LIMIT = 3;
  public function __construct(int $x) { $this->x = $x; }
  public function area(): float { return $this->helper($this->x) + Helper::compute(1); }
  private static function helper(int $n): float { return $n; }
}
trait Greets { public function hi() {} }
enum Color { case Red; case Green; }
function top($n) { return top($n - 1); }
