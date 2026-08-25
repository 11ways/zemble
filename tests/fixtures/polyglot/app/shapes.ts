export interface Shape { area(): number }

export class Circle implements Shape {
  constructor(private radius: number) {}
  area(): number { return scale(3 * this.radius * this.radius); }
}

export function scale(value: number): number { return value * 2; }

export function largest(shapes: Shape[]): Shape | undefined {
  let best: Shape | undefined;
  for (const shape of shapes) {
    if (best === undefined || shape.area() > best.area()) { best = shape; }
  }
  return best;
}
