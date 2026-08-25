using System;
namespace App.Core {
  public interface IShape { double Area(); }
  public abstract class Base { }
  public class Point : Base, IShape {
    private int x;
    public int Count { get; set; }
    public Point(int x) { this.x = x; }
    public double Area() { return Helper.Compute(x); }
    public static Point Make() => new Point(1);
  }
  public struct Vec { public int X; }
  public enum Color { Red, Green }
  public record Person(string Name);
  public static class Helper { public static double Compute(int n) { return n; } }
}
