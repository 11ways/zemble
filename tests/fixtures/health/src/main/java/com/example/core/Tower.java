package com.example.core;

/** Seven classes in one chain: the last has six ancestors, one past the bar. */
public class Tower {

    public static class Level1 {}

    public static class Level2 extends Level1 {}

    public static class Level3 extends Level2 {}

    public static class Level4 extends Level3 {}

    public static class Level5 extends Level4 {}

    public static class Level6 extends Level5 {}

    public static class Level7 extends Level6 {}
}
