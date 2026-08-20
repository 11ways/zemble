package be.elevenways.hawkeye.generated.zenitwidget;

import be.elevenways.domino.common.DominoElement;
import be.elevenways.hawkeye.common.Hawkeye;
import be.elevenways.hawkeye.common.render.RenderContext;
import be.elevenways.hawkeye.common.template.CompiledTemplate;
import be.elevenways.hawkeye.common.template.DeclaredVariable;
import be.elevenways.hawkeye.common.template.Template;
import be.elevenways.protoblast.common.Blast;
import be.elevenways.protoblast.common.holder.Ref;
import be.elevenways.protoblast.common.registry.Identifier;
import be.elevenways.zenit.common.text.MarkdownTemplateFunctions;
import be.elevenways.zenit.widget.common.display.WidgetDisplayFunctions;
import java.lang.Object;
import java.lang.Override;
import java.lang.String;
import java.lang.SuppressWarnings;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import be.elevenways.hawkeye.common.sourcemap.SourceMapRegistry;

public final class Tpl_WidgetDisplayMarkdown extends CompiledTemplate {
    private static final Identifier TEMPLATE_ID = Identifier.of("zenitwidget", "widget-display/markdown");

    private static final List<DeclaredVariable> DECLARED_VARIABLES = List.of(new DeclaredVariable("widget", new be.elevenways.hawkeye.common.parser.ast.TypeAnnotation("WidgetInstance")));

    static {
        HawkeyeClassSerializers.init();
        HawkeyeCustomElementRegistrations.init();
        HawkeyeDeclaredClassRegistrations.init();
        Blast.ensureAutoLoaded();
        be.elevenways.hawkeye.common.sourcemap.SourceMapRegistry.registerFromCompact("widget-display/markdown", "be.elevenways.hawkeye.generated.zenitwidget.Tpl_WidgetDisplayMarkdown", new int[]{61, 6, 73, 8});
        Hawkeye.ALL_TEMPLATES.add(TEMPLATE_ID, Tpl_WidgetDisplayMarkdown::new);
    }

    public Tpl_WidgetDisplayMarkdown() {
        super();
    }

    @Override
    public Identifier getIdentifier() {
        return TEMPLATE_ID;
    }

    @Override
    public Supplier<Template> getFactory() {
        return Tpl_WidgetDisplayMarkdown::new;
    }

    /**
     * No-op; calling this forces class loading to run the registering static initializer.
     */
    public static void init() {
    }

    @Override
    public void renderRoot(RenderContext context) {
        this.validateContext(context);
        this.printUnsafe(context, " ");
        this.branch1(context);
        this.printUnsafe(context, " ");
        // @hwk:6
        DominoElement element1 = this.openElement(context, "div", "widget widget-markdown");
        this.populatedOpeningTag(context, element1);
        this.branch2(context);
        this.closeElement(context, element1);
    }

    private void branch1(RenderContext context) {
    }

    private void branch2(RenderContext context) {
        this.printUnsafe(context, " ");
        // @hwk:8
        @SuppressWarnings("unchecked")
        Ref<Object> expr_2 = (Ref<Object>) context.getVariableRef("widget");
        Object unwrapped_3 = RenderContext.unwrapCurrent(expr_2);
        Object expr_4 = this.getValueFromPath(context, unwrapped_3, "config");
        Object unwrapped_5 = RenderContext.unwrapCurrent(expr_4);
        Object result_6 = (Object) WidgetDisplayFunctions.localizedConfig(context, (Map) unwrapped_5, (String) "body");
        Object unwrapped_7 = RenderContext.unwrapCurrent(result_6);
        String result_8 = (String) MarkdownTemplateFunctions.render((Object) unwrapped_7);
        this.printUnsafe(context, result_8);
        this.printUnsafe(context, " ");
    }

    @Override
    public List<DeclaredVariable> getDeclaredVariables() {
        return DECLARED_VARIABLES;
    }
}
