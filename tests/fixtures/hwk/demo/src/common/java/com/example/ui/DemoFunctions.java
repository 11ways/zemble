package com.example.ui;

import com.example.annotation.HawkeyeFunction;

/** Template functions the demo templates call. */
public class DemoFunctions {

    @HawkeyeFunction(name = "label", namespace = "Demo", description = "The display label of a row")
    public static String label(String target) {
        return target == null ? "" : target;
    }

    /** A global function: templates call it as a bare `t(...)`. */
    @HawkeyeFunction(name = "t")
    public static String translate(String key) {
        return key;
    }

    /** Not exposed to templates at all, and so never the target of a template call. */
    public static String label(int index, String target) {
        return index + target;
    }
}
