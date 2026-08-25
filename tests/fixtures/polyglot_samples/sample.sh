#!/bin/bash
source lib.sh
area() { helper "$1"; echo done; }
function helper { echo "$1"; }
area 3
