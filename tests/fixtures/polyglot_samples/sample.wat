(module
  (type $t (func (param i32) (result i32)))
  (func $helper (param $n i32) (result i32) (i32.add (local.get $n) (i32.const 1)))
  (func $area (export "area") (param $x i32) (result i32) (call $helper (local.get $x)))
  (global $limit i32 (i32.const 3))
  (memory 1))
