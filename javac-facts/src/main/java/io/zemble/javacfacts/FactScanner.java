package io.zemble.javacfacts;

import com.sun.source.tree.BlockTree;
import com.sun.source.tree.ClassTree;
import com.sun.source.tree.CompilationUnitTree;
import com.sun.source.tree.MemberReferenceTree;
import com.sun.source.tree.MethodInvocationTree;
import com.sun.source.tree.MethodTree;
import com.sun.source.tree.NewClassTree;
import com.sun.source.tree.Tree;
import com.sun.source.tree.VariableTree;
import com.sun.source.util.SourcePositions;
import com.sun.source.util.TreePath;
import com.sun.source.util.TreePathScanner;
import com.sun.source.util.Trees;

import javax.lang.model.element.AnnotationMirror;
import javax.lang.model.element.AnnotationValue;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.TypeKind;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Walks one attributed top-level class and appends its facts to a buffer. */
final class FactScanner extends TreePathScanner<Void, Void> {

    private final Trees trees;
    private final Elements elements;
    private final Types types;
    private final Refs refs;
    private final SourcePositions positions;
    private final CompilationUnitTree unit;
    private final String path;
    private final StringBuilder out;

    private final Deque<TypeElement> classes = new ArrayDeque<>();
    private final Deque<String> callers = new ArrayDeque<>();

    FactScanner(Trees trees, Elements elements, Types types, Refs refs,
                CompilationUnitTree unit, String path, StringBuilder out) {
        this.trees = trees;
        this.elements = elements;
        this.types = types;
        this.refs = refs;
        this.positions = trees.getSourcePositions();
        this.unit = unit;
        this.path = path;
        this.out = out;
    }

    @Override
    public Void visitClass(ClassTree node, Void unused) {
        Element element = trees.getElement(getCurrentPath());
        if (!(element instanceof TypeElement type)) {
            return super.visitClass(node, unused);
        }

        String ref = refs.type(type);
        symbol(ref, kindOf(type), node);
        annotations(ref, type);
        hierarchy(type, ref);

        classes.push(type);
        callers.push("");
        try {
            return super.visitClass(node, unused);
        } finally {
            callers.pop();
            classes.pop();
        }
    }

    @Override
    public Void visitMethod(MethodTree node, Void unused) {
        Element element = trees.getElement(getCurrentPath());
        if (!(element instanceof ExecutableElement method)) {
            return super.visitMethod(node, unused);
        }

        String ref = refs.method(method);
        boolean constructor = method.getKind() == ElementKind.CONSTRUCTOR;
        symbol(ref, constructor ? "constructor" : "method", node);
        annotations(ref, method);
        if (!constructor) {
            overrides(method, ref);
        }

        callers.push(ref);
        try {
            return super.visitMethod(node, unused);
        } finally {
            callers.pop();
        }
    }

    @Override
    public Void visitVariable(VariableTree node, Void unused) {
        Element element = trees.getElement(getCurrentPath());
        if (!(element instanceof VariableElement variable)
                || (element.getKind() != ElementKind.FIELD && element.getKind() != ElementKind.ENUM_CONSTANT)) {
            return super.visitVariable(node, unused);
        }

        String ref = refs.field(variable);
        symbol(ref, "field", node);
        annotations(ref, variable);

        callers.push(initializerRef(variable.getModifiers().contains(Modifier.STATIC)));
        try {
            return super.visitVariable(node, unused);
        } finally {
            callers.pop();
        }
    }

    @Override
    public Void visitBlock(BlockTree node, Void unused) {
        TreePath parent = getCurrentPath().getParentPath();
        boolean initializer = parent != null && parent.getLeaf() instanceof ClassTree;
        if (!initializer) {
            return super.visitBlock(node, unused);
        }

        callers.push(initializerRef(node.isStatic()));
        try {
            return super.visitBlock(node, unused);
        } finally {
            callers.pop();
        }
    }

    @Override
    public Void visitMethodInvocation(MethodInvocationTree node, Void unused) {
        Element element = trees.getElement(getCurrentPath());
        if (!(element instanceof ExecutableElement)) {
            element = trees.getElement(new TreePath(getCurrentPath(), node.getMethodSelect()));
        }
        if (element instanceof ExecutableElement method) {
            call(refs.method(method), node);
        }
        return super.visitMethodInvocation(node, unused);
    }

    @Override
    public Void visitNewClass(NewClassTree node, Void unused) {
        Element element = trees.getElement(getCurrentPath());
        if (element instanceof ExecutableElement constructor) {
            call(refs.method(constructor), node);
        }
        return super.visitNewClass(node, unused);
    }

    @Override
    public Void visitMemberReference(MemberReferenceTree node, Void unused) {
        Element element = trees.getElement(getCurrentPath());
        if (element instanceof ExecutableElement method) {
            call(refs.method(method), node);
        }
        return super.visitMemberReference(node, unused);
    }

    private String initializerRef(boolean isStatic) {
        TypeElement owner = classes.peek();
        if (owner == null) {
            return "";
        }
        return isStatic ? refs.staticInitializer(owner) : refs.instanceInitializer(owner);
    }

    private String currentCaller() {
        String caller = callers.peek();
        if (caller != null && !caller.isEmpty()) {
            return caller;
        }
        TypeElement owner = classes.peek();
        return owner == null ? null : refs.instanceInitializer(owner);
    }

    private void symbol(String ref, String kind, Tree node) {
        out.append('{');
        Json.field(out, "t", "symbol");
        out.append(',');
        Json.field(out, "ref", ref);
        out.append(',');
        Json.field(out, "path", path);
        out.append(',');
        Json.raw(out, "line", Long.toString(lineOf(node)));
        out.append(',');
        Json.field(out, "kind", kind);
        out.append("}\n");
    }

    private void call(String to, Tree node) {
        String from = currentCaller();
        if (from == null) {
            return;
        }
        out.append('{');
        Json.field(out, "t", "call");
        out.append(',');
        Json.field(out, "from", from);
        out.append(',');
        Json.field(out, "to", to);
        out.append(',');
        Json.field(out, "path", path);
        out.append(',');
        Json.raw(out, "line", Long.toString(lineOf(node)));
        out.append("}\n");
    }

    private void edge(String kind, String from, String to) {
        out.append('{');
        Json.field(out, "t", kind);
        out.append(',');
        Json.field(out, "from", from);
        out.append(',');
        Json.field(out, "to", to);
        out.append("}\n");
    }

    private void hierarchy(TypeElement type, String ref) {
        boolean isInterface = type.getKind() == ElementKind.INTERFACE
                || type.getKind() == ElementKind.ANNOTATION_TYPE;

        TypeMirror superclass = type.getSuperclass();
        if (superclass.getKind() == TypeKind.DECLARED) {
            TypeElement superElement = (TypeElement) ((DeclaredType) superclass).asElement();
            if (!"java.lang.Object".contentEquals(superElement.getQualifiedName())) {
                edge("extends", ref, refs.type(superElement));
            }
        }

        for (TypeMirror implemented : type.getInterfaces()) {
            if (implemented.getKind() != TypeKind.DECLARED) {
                continue;
            }
            TypeElement implementedElement = (TypeElement) ((DeclaredType) implemented).asElement();
            edge(isInterface ? "extends" : "implements", ref, refs.type(implementedElement));
        }
    }

    private void overrides(ExecutableElement method, String ref) {
        TypeElement owner = (TypeElement) method.getEnclosingElement();
        Set<String> seen = new LinkedHashSet<>();
        for (TypeMirror supertype : types.directSupertypes(owner.asType())) {
            collectOverridden(method, owner, supertype, seen);
        }
        for (String overridden : seen) {
            edge("override", ref, overridden);
        }
    }

    private void collectOverridden(ExecutableElement method, TypeElement owner, TypeMirror supertype, Set<String> seen) {
        if (supertype.getKind() != TypeKind.DECLARED) {
            return;
        }
        TypeElement element = (TypeElement) ((DeclaredType) supertype).asElement();
        boolean found = false;
        for (Element member : element.getEnclosedElements()) {
            if (member.getKind() != ElementKind.METHOD) {
                continue;
            }
            ExecutableElement candidate = (ExecutableElement) member;
            if (!candidate.getSimpleName().contentEquals(method.getSimpleName())) {
                continue;
            }
            if (elements.overrides(method, candidate, owner)) {
                seen.add(refs.method(candidate));
                found = true;
            }
        }
        if (found) {
            return;
        }
        for (TypeMirror next : types.directSupertypes(supertype)) {
            collectOverridden(method, owner, next, seen);
        }
    }

    private void annotations(String ref, Element element) {
        for (AnnotationMirror mirror : element.getAnnotationMirrors()) {
            TypeElement annotationType = (TypeElement) mirror.getAnnotationType().asElement();
            Map<String, Object> args = new LinkedHashMap<>();
            for (Map.Entry<? extends ExecutableElement, ? extends AnnotationValue> entry
                    : mirror.getElementValues().entrySet()) {
                Object value = constantOf(entry.getValue());
                if (value != null) {
                    args.put(entry.getKey().getSimpleName().toString(), value);
                }
            }

            out.append('{');
            Json.field(out, "t", "annotation");
            out.append(',');
            Json.field(out, "ref", ref);
            out.append(',');
            Json.field(out, "name", refs.type(annotationType));
            out.append(',');
            Json.string(out, "args");
            out.append(':');
            Json.object(out, args);
            out.append("}\n");
        }
    }

    /** Normalises an annotation argument to a JSON-safe constant, or null when it is not one. */
    private Object constantOf(AnnotationValue annotationValue) {
        Object value = annotationValue.getValue();
        if (value instanceof String || value instanceof Boolean || value instanceof Number) {
            return value;
        }
        if (value instanceof Character character) {
            return String.valueOf(character);
        }
        if (value instanceof VariableElement enumConstant) {
            return enumConstant.getSimpleName().toString();
        }
        if (value instanceof List<?> items) {
            List<Object> normalised = new ArrayList<>();
            for (Object item : items) {
                Object element = item instanceof AnnotationValue nested ? constantOf(nested) : null;
                if (element == null) {
                    return null;
                }
                normalised.add(element);
            }
            return normalised;
        }
        return null;
    }

    private static String kindOf(TypeElement type) {
        return switch (type.getKind()) {
            case INTERFACE -> "interface";
            case ENUM -> "enum";
            case RECORD -> "record";
            case ANNOTATION_TYPE -> "annotation";
            default -> "class";
        };
    }

    private long lineOf(Tree node) {
        long position = positions.getStartPosition(unit, node);
        if (position < 0) {
            return 0;
        }
        return unit.getLineMap().getLineNumber(position);
    }
}
