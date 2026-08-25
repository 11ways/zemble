import { thing } from './mod';
export interface Shape extends Base { area(): number; name: string }
export abstract class Foo<T> extends Bar implements Shape, Other {
  private count: number = 0;
  constructor(private a: T) { super(); }
  public run(b: string, c?: number): void { this.helper(b); thing.call(c); }
  abstract area(): number;
}
export enum Color { Red, Green = 2 }
export type Alias = string | number;
export namespace NS { export function inner(): void {} }
export function top(n: number): number { return top(n - 1); }
export const arrow = (a: number): number => a + 1;
