package io.zemble.javacfacts;

import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.ExecutableElement;
import javax.lang.model.element.Name;
import javax.lang.model.element.NestingKind;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;
import javax.lang.model.type.ArrayType;
import javax.lang.model.type.DeclaredType;
import javax.lang.model.type.ExecutableType;
import javax.lang.model.type.TypeMirror;
import javax.lang.model.util.Elements;
import javax.lang.model.util.Types;

/**
 * Renders the zemble graph-facts ref grammar.
 *
 * <p>Nested types use their source-style dotted name; anonymous and local types (and anything
 * nested inside one) fall back to javac's flat binary name, so {@code pkg.Outer$1} identifies an
 * anonymous class exactly as the class file does.
 */
final class Refs {

    private final Elements elements;
    private final Types types;

    Refs(Elements elements, Types types) {
        this.elements = elements;
        this.types = types;
    }

    String type(TypeElement element) {
        if (usesFlatName(element)) {
            return elements.getBinaryName(element).toString();
        }
        Name qualified = element.getQualifiedName();
        if (qualified.length() > 0) {
            return qualified.toString();
        }
        return elements.getBinaryName(element).toString();
    }

    /** Erases the type and renders it fully qualified, with {@code []} for arrays. */
    String typeName(TypeMirror mirror) {
        TypeMirror erased = types.erasure(mirror);
        return switch (erased.getKind()) {
            case ARRAY -> typeName(((ArrayType) erased).getComponentType()) + "[]";
            case DECLARED -> type((TypeElement) ((DeclaredType) erased).asElement());
            default -> erased.toString();
        };
    }

    String method(ExecutableElement element) {
        Element owner = element.getEnclosingElement();
        String ownerRef = owner instanceof TypeElement typeElement
                ? type(typeElement)
                : String.valueOf(owner);
        StringBuilder ref = new StringBuilder(ownerRef);
        ref.append('#');
        ref.append(element.getKind() == ElementKind.CONSTRUCTOR ? "<init>" : element.getSimpleName());
        ref.append('(');
        TypeMirror erased = types.erasure(element.asType());
        boolean first = true;
        if (erased instanceof ExecutableType executable) {
            for (TypeMirror parameter : executable.getParameterTypes()) {
                if (!first) {
                    ref.append(',');
                }
                first = false;
                ref.append(typeName(parameter));
            }
        } else {
            for (VariableElement parameter : element.getParameters()) {
                if (!first) {
                    ref.append(',');
                }
                first = false;
                ref.append(typeName(parameter.asType()));
            }
        }
        ref.append(')');
        return ref.toString();
    }

    String field(VariableElement element) {
        Element owner = element.getEnclosingElement();
        String ownerRef = owner instanceof TypeElement typeElement
                ? type(typeElement)
                : String.valueOf(owner);
        return ownerRef + "#" + element.getSimpleName();
    }

    String staticInitializer(TypeElement owner) {
        return type(owner) + "#<clinit>()";
    }

    String instanceInitializer(TypeElement owner) {
        return type(owner) + "#<instance-init>()";
    }

    private static boolean usesFlatName(TypeElement element) {
        Element current = element;
        while (current instanceof TypeElement typeElement) {
            NestingKind nesting = typeElement.getNestingKind();
            if (nesting == NestingKind.ANONYMOUS || nesting == NestingKind.LOCAL) {
                return true;
            }
            current = typeElement.getEnclosingElement();
        }
        return false;
    }
}
