function Get-Area { param($x) Get-Helper $x }
function Get-Helper($n) { return $n + 1 }
class Point { [int]$x; Point($x) { $this.x = $x } [int]Area() { return Get-Helper $this.x } }
