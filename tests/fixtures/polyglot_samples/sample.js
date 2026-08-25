import { thing } from './mod';
const x = 1;
export class Foo extends Bar {
  static count = 0;
  #priv = 2;
  constructor(a) { super(a); this.a = a; }
  get value() { return this.a; }
  run(b, c) { this.helper(b); thing.call(c); new Foo(1); }
}
export function top(n) { return top(n - 1); }
const arrow = (a) => a + 1;
function* gen() {}
module.exports = { top };
