package com.example.ui;

import com.example.annotation.HawkeyeCustomElement;

/** A hand-written custom element, registered under a literal tag. */
@HawkeyeCustomElement(tag = "demo-widget", name = "DemoWidget")
public final class DemoWidgetElement {

    public String render() {
        return "<demo-widget></demo-widget>";
    }
}
