package com.example.ui;

import com.example.annotation.HawkeyeFunction;

/** The template functions `widget-display/markdown.hwk` calls. */
public class WidgetFunctions {

    @HawkeyeFunction(name = "render", namespace = "Markdown", description = "Safe markdown rendering")
    public static String render(String body) {
        return body == null ? "" : body;
    }

    @HawkeyeFunction(name = "localizedConfig", namespace = "Widget", description = "One localized config value")
    public static String localizedConfig(Object config, String key) {
        return key;
    }
}
