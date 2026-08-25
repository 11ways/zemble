"""One :class:`LanguageSpec` per bundled grammar: the home of every grammar-specific fact.

Java and Hawkeye templates keep their hand-written graph extractors and Zig its hand-written
duplication profile, so they have no entry here. Every other grammar `semble_grammars` ships
is listed, including the ones with nothing to declare (a stylesheet, a Dockerfile), so that
"no spec" always means "no grammar" and never "forgot to add one".
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import PurePosixPath

from zemble.graph.model import SymbolKind
from zemble.index.files import detect_language
from zemble.languages.spec import CallRule, LanguageSpec, Role, Rule
from zemble.languages.visibility import Visibility

_CLASS = SymbolKind.CLASS
_INTERFACE = SymbolKind.INTERFACE
_ENUM = SymbolKind.ENUM
_RECORD = SymbolKind.RECORD
_STRUCT = SymbolKind.STRUCT
_TYPE = SymbolKind.TYPE
_MODULE = SymbolKind.MODULE

_C_LIKE_MODIFIERS = frozenset(
    {
        "public",
        "private",
        "protected",
        "internal",
        "static",
        "abstract",
        "final",
        "async",
        "export",
        "default",
        "const",
        "override",
        "virtual",
        "inline",
        "extern",
        "unsafe",
        "readonly",
        "sealed",
        "partial",
        "volatile",
        "declare",
        "constexpr",
        "explicit",
        "friend",
        "mutable",
        "new",
        "unsafe",
        "required",
        "ref",
        "out",
        "in",
        "params",
        "get",
        "set",
    }
)

_JS_BINDINGS = (
    ("formal_parameters", "@identifier*"),
    ("variable_declarator", "name"),
    ("assignment_pattern", "left"),
    ("rest_pattern", "@identifier"),
    ("arrow_function", "parameter"),
    ("catch_clause", "parameter"),
    ("for_in_statement", "left"),
    ("object_pattern", "@shorthand_property_identifier_pattern*"),
    ("array_pattern", "@identifier*"),
)

_JS_RULES = (
    Rule(("class_declaration",), Role.TYPE, _CLASS, supertypes=("@class_heritage",)),
    Rule(
        ("variable_declarator",),
        Role.CALLABLE,
        body="value/body",
        params="value/parameters",
        require=("value", "arrow_function"),
    ),
    Rule(
        ("variable_declarator",),
        Role.CALLABLE,
        body="value/body",
        params="value/parameters",
        require=("value", "function_expression"),
    ),
    Rule(("method_definition",), Role.CALLABLE, params="parameters"),
    Rule(("function_declaration", "generator_function_declaration"), Role.CALLABLE, params="parameters"),
    Rule(("field_definition",), Role.FIELD, name="property", body=None),
    Rule(("variable_declarator",), Role.FIELD, body=None, scope="member"),
    Rule(
        ("import_statement",),
        Role.IMPORT,
        name="source",
        names="@import_clause/@named_imports/@import_specifier*|@import_clause/@identifier",
        body=None,
    ),
)

_JS_CALLS = (
    CallRule(("call_expression",), callee="function", arguments="arguments"),
    CallRule(("new_expression",), callee="constructor", arguments="arguments", is_new=True),
)

_JS_BUILTINS = frozenset(
    {
        "string",
        "number",
        "boolean",
        "any",
        "void",
        "unknown",
        "never",
        "object",
        "symbol",
        "undefined",
        "null",
        "bigint",
        "Promise",
        "Array",
        "Record",
        "Partial",
        "Readonly",
        "Map",
        "Set",
        "Date",
        "Error",
        "Function",
        "Object",
        "String",
        "Number",
        "Boolean",
        "this",
    }
)

JAVASCRIPT = LanguageSpec(
    language="javascript",
    family="js",
    rules=_JS_RULES,
    calls=_JS_CALLS,
    wrappers={"export_statement": "declaration"},
    member_access={"member_expression": ("object", "property")},
    self_names=frozenset({"this"}),
    builtin_types=_JS_BUILTINS,
    annotation_kinds=frozenset({"decorator"}),
    modifier_words=frozenset({"static", "async", "export", "default", "get", "set"}),
    block_kinds=frozenset({"statement_block"}),
    binding_paths=_JS_BINDINGS,
    constructor_names=frozenset({"constructor"}),
    test_file_patterns=("*.test.js", "*.spec.js", "*.test.jsx", "*.spec.jsx", "*.test.mjs", "*.spec.mjs"),
)

_TS_RULES = (
    Rule(
        ("class_declaration", "abstract_class_declaration"),
        Role.TYPE,
        _CLASS,
        supertypes=("@class_heritage/@extends_clause", "@class_heritage/@implements_clause"),
    ),
    Rule(("interface_declaration",), Role.TYPE, _INTERFACE, supertypes=("@extends_type_clause",)),
    Rule(("enum_declaration",), Role.TYPE, _ENUM),
    Rule(("type_alias_declaration",), Role.TYPE, _TYPE, body=None),
    Rule(("internal_module", "module"), Role.MODULE),
    Rule(("enum_assignment",), Role.CONSTANT, body=None),
    Rule(("property_identifier",), Role.CONSTANT, name=".", body=None, require=("..", "enum_body")),
    Rule(
        ("variable_declarator",),
        Role.CALLABLE,
        body="value/body",
        params="value/parameters",
        require=("value", "arrow_function"),
    ),
    Rule(
        ("variable_declarator",),
        Role.CALLABLE,
        body="value/body",
        params="value/parameters",
        require=("value", "function_expression"),
    ),
    Rule(("method_definition",), Role.CALLABLE, params="parameters", return_type="return_type"),
    Rule(
        ("method_signature", "abstract_method_signature"),
        Role.CALLABLE,
        params="parameters",
        body=None,
        return_type="return_type",
    ),
    Rule(
        ("function_declaration", "generator_function_declaration", "function_signature"),
        Role.CALLABLE,
        params="parameters",
        return_type="return_type",
    ),
    Rule(("public_field_definition", "property_signature"), Role.FIELD, body=None),
    Rule(("variable_declarator",), Role.FIELD, body=None, scope="member"),
    Rule(
        ("import_statement",),
        Role.IMPORT,
        name="source",
        names="@import_clause/@named_imports/@import_specifier*|@import_clause/@identifier",
        body=None,
    ),
)

_TS_BINDINGS = (
    *_JS_BINDINGS,
    ("required_parameter", "pattern"),
    ("optional_parameter", "pattern"),
)

TYPESCRIPT = LanguageSpec(
    language="typescript",
    family="js",
    rules=_TS_RULES,
    calls=_JS_CALLS,
    wrappers={"export_statement": "declaration", "ambient_declaration": "@*"},
    member_access={"member_expression": ("object", "property")},
    self_names=frozenset({"this"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=_JS_BUILTINS,
    annotation_kinds=frozenset({"decorator"}),
    modifier_kinds=frozenset({"accessibility_modifier", "override_modifier"}),
    modifier_words=frozenset({"static", "async", "export", "default", "abstract", "readonly", "declare", "get", "set"}),
    block_kinds=frozenset({"statement_block"}),
    binding_paths=_TS_BINDINGS,
    constructor_names=frozenset({"constructor"}),
    test_file_patterns=("*.test.ts", "*.spec.ts", "*.test.tsx", "*.spec.tsx", "*.test.mts"),
)

TSX = LanguageSpec(
    language="tsx",
    family="js",
    rules=_TS_RULES,
    calls=_JS_CALLS,
    wrappers=TYPESCRIPT.wrappers,
    member_access=TYPESCRIPT.member_access,
    self_names=frozenset({"this"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=_JS_BUILTINS,
    annotation_kinds=frozenset({"decorator"}),
    modifier_kinds=TYPESCRIPT.modifier_kinds,
    modifier_words=TYPESCRIPT.modifier_words,
    block_kinds=frozenset({"statement_block"}),
    binding_paths=_TS_BINDINGS,
    constructor_names=frozenset({"constructor"}),
    test_file_patterns=TYPESCRIPT.test_file_patterns,
)

PYTHON = LanguageSpec(
    language="python",
    family="python",
    rules=(
        Rule(("class_definition",), Role.TYPE, _CLASS, supertypes=("superclasses",)),
        Rule(("function_definition",), Role.CALLABLE, params="parameters", return_type="return_type"),
        Rule(
            ("expression_statement",),
            Role.FIELD,
            name="@assignment/left",
            body=None,
            require=("@assignment/left", "identifier"),
            scope="member",
        ),
        Rule(("import_statement",), Role.IMPORT, name="@dotted_name|@aliased_import/name", body=None),
        Rule(("import_from_statement",), Role.IMPORT, name="module_name", names="name+", body=None),
    ),
    calls=(CallRule(("call",), callee="function", arguments="arguments"),),
    wrappers={"decorated_definition": "definition"},
    member_access={"attribute": ("object", "attribute")},
    self_names=frozenset({"self", "cls"}),
    self_parameters=frozenset({"self", "cls"}),
    type_ref_kinds=frozenset({"type"}),
    builtin_types=frozenset(
        {
            "int",
            "str",
            "float",
            "bool",
            "bytes",
            "list",
            "dict",
            "set",
            "tuple",
            "frozenset",
            "None",
            "object",
            "Any",
            "Optional",
            "Union",
            "List",
            "Dict",
            "Set",
            "Tuple",
            "Callable",
            "Iterable",
            "Iterator",
            "Sequence",
            "Mapping",
            "type",
            "Self",
            "complex",
            "bytearray",
        }
    ),
    annotation_kinds=frozenset({"decorator"}),
    modifier_words=frozenset({"async"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameters", "@identifier*"),
        ("default_parameter", "name"),
        ("typed_parameter", "@identifier"),
        ("typed_default_parameter", "name"),
        ("list_splat_pattern", "@identifier"),
        ("dictionary_splat_pattern", "@identifier"),
        ("lambda_parameters", "@identifier*"),
        ("assignment", "left"),
        ("augmented_assignment", "left"),
        ("for_statement", "left"),
        ("for_in_clause", "left"),
        ("as_pattern", "alias"),
        ("named_expression", "name"),
        ("with_item", "value/alias"),
        ("pattern_list", "@identifier*"),
        ("tuple_pattern", "@identifier*"),
    ),
    constructor_names=frozenset({"__init__", "__new__"}),
    capitalized_call_is_new=True,
    underscore_is_private=True,
    test_file_patterns=("test_*.py", "*_test.py", "conftest.py"),
)

STARLARK = LanguageSpec(
    language="starlark",
    family="starlark",
    rules=(
        Rule(("function_definition",), Role.CALLABLE, params="parameters"),
        Rule(
            ("expression_statement",),
            Role.IMPORT,
            name="@call/arguments/@string",
            names="@call/arguments/@string*",
            body=None,
            when=("@call/function", frozenset({"load"})),
        ),
        Rule(
            ("expression_statement",),
            Role.FIELD,
            name="@assignment/left",
            body=None,
            require=("@assignment/left", "identifier"),
            scope="member",
        ),
    ),
    calls=(CallRule(("call",), callee="function", arguments="arguments"),),
    member_access={"attribute": ("object", "attribute")},
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameters", "@identifier*"),
        ("default_parameter", "name"),
        ("assignment", "left"),
        ("for_statement", "left"),
        ("lambda_parameters", "@identifier*"),
    ),
    underscore_is_private=True,
    test_file_patterns=("*_test.bzl",),
)

GO = LanguageSpec(
    language="go",
    family="go",
    rules=(
        Rule(("package_clause",), Role.PACKAGE, name="@package_identifier", body=None),
        Rule(("import_spec",), Role.IMPORT, name="path", body=None),
        Rule(("type_spec",), Role.TYPE, _STRUCT, body="type/@field_declaration_list", require=("type", "struct_type")),
        Rule(("type_spec",), Role.TYPE, _INTERFACE, body="type", require=("type", "interface_type")),
        Rule(("type_spec", "type_alias"), Role.TYPE, _TYPE, body=None),
        Rule(("method_elem",), Role.CALLABLE, params="parameters", body=None, return_type="result"),
        Rule(("field_declaration",), Role.FIELD, names="name+", body=None),
        Rule(
            ("method_declaration",),
            Role.CALLABLE,
            params="parameters",
            owner="receiver/**type_identifier",
            return_type="result",
        ),
        Rule(("function_declaration",), Role.CALLABLE, params="parameters", return_type="result"),
        Rule(("const_spec", "var_spec"), Role.FIELD, names="name+", body=None, scope="member"),
    ),
    calls=(
        CallRule(("call_expression",), callee="function", arguments="arguments"),
        CallRule(("composite_literal",), callee="type", arguments="body", is_new=True),
    ),
    member_access={"selector_expression": ("operand", "field")},
    self_names=frozenset(),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=frozenset(
        {
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "uintptr",
            "float32",
            "float64",
            "complex64",
            "complex128",
            "string",
            "bool",
            "byte",
            "rune",
            "error",
            "any",
            "comparable",
        }
    ),
    block_kinds=frozenset({"statement_list"}),
    binding_paths=(
        ("parameter_declaration", "name+"),
        ("variadic_parameter_declaration", "name"),
        ("short_var_declaration", "left/@identifier*"),
        ("var_spec", "name+"),
        ("const_spec", "name+"),
        ("range_clause", "left/@identifier*"),
    ),
    capitalized_is_public=True,
    default_visibility=Visibility.PACKAGE,
    test_file_patterns=("*_test.go",),
)

RUST = LanguageSpec(
    language="rust",
    family="rust",
    rules=(
        Rule(("use_declaration",), Role.IMPORT, name="argument", body=None),
        Rule(("mod_item",), Role.MODULE),
        Rule(("struct_item", "union_item"), Role.TYPE, _STRUCT),
        Rule(("enum_item",), Role.TYPE, _ENUM),
        Rule(("trait_item",), Role.TYPE, _INTERFACE),
        Rule(("type_item",), Role.TYPE, _TYPE, body=None),
        Rule(("impl_item",), Role.EXTENSION, owner="type", trait="trait"),
        Rule(("function_item",), Role.CALLABLE, params="parameters", return_type="return_type"),
        Rule(("function_signature_item",), Role.CALLABLE, params="parameters", body=None, return_type="return_type"),
        Rule(("field_declaration",), Role.FIELD, body=None),
        Rule(("enum_variant",), Role.CONSTANT, body=None),
        Rule(("const_item", "static_item"), Role.FIELD, body=None, scope="member"),
    ),
    calls=(
        CallRule(("call_expression",), callee="function", arguments="arguments"),
        CallRule(("struct_expression",), callee="name", arguments="body", is_new=True),
    ),
    member_access={"field_expression": ("value", "field"), "scoped_identifier": ("path", "name")},
    self_names=frozenset({"self", "Self"}),
    self_parameters=frozenset({"self_parameter"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=frozenset(
        {
            "i8",
            "i16",
            "i32",
            "i64",
            "i128",
            "isize",
            "u8",
            "u16",
            "u32",
            "u64",
            "u128",
            "usize",
            "f32",
            "f64",
            "bool",
            "char",
            "str",
            "String",
            "Vec",
            "Option",
            "Result",
            "Box",
            "Rc",
            "Arc",
            "HashMap",
            "HashSet",
            "Self",
        }
    ),
    annotation_kinds=frozenset({"attribute_item"}),
    modifier_kinds=frozenset({"visibility_modifier"}),
    modifier_words=frozenset({"const", "async", "unsafe", "extern", "static", "mut"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameter", "pattern"),
        ("let_declaration", "pattern"),
        ("closure_parameters", "@identifier*"),
        ("for_expression", "pattern"),
        ("tuple_pattern", "@identifier*"),
    ),
    default_visibility=Visibility.PRIVATE,
)

_C_RULES = (
    Rule(("preproc_include",), Role.IMPORT, name="path", body=None),
    Rule(("preproc_def",), Role.FIELD, body=None, scope="top"),
    Rule(("preproc_function_def",), Role.CALLABLE, params="parameters", body="value"),
    Rule(
        ("function_definition",),
        Role.CALLABLE,
        name="declarator~function_declarator/declarator*",
        params="declarator~function_declarator/parameters",
        require=("declarator~function_declarator", "function_declarator"),
        return_type="type",
    ),
    Rule(
        ("type_definition",),
        Role.TYPE,
        _STRUCT,
        name="declarator",
        body="type/body",
        require=("type/body", "field_declaration_list"),
    ),
    Rule(
        ("type_definition",),
        Role.TYPE,
        _ENUM,
        name="declarator",
        body="type/body",
        require=("type/body", "enumerator_list"),
    ),
    Rule(("type_definition",), Role.TYPE, _TYPE, name="declarator", body=None),
    Rule(("struct_specifier", "union_specifier"), Role.TYPE, _STRUCT, require=("body", "field_declaration_list")),
    Rule(("enum_specifier",), Role.TYPE, _ENUM, require=("body", "enumerator_list")),
    Rule(("enumerator",), Role.CONSTANT, body=None),
    Rule(("declaration",), Role.SKIP, require=("declarator~function_declarator", "function_declarator")),
    Rule(("declaration",), Role.FIELD, name="declarator*", body=None, scope="member"),
    Rule(("field_declaration",), Role.FIELD, name="declarator*", body=None),
)

_C_BINDINGS = (
    ("parameter_declaration", "declarator*"),
    ("declaration", "declarator*"),
    ("init_declarator", "declarator*"),
)

_C_BUILTINS = frozenset(
    {
        "size_t",
        "ssize_t",
        "int8_t",
        "int16_t",
        "int32_t",
        "int64_t",
        "uint8_t",
        "uint16_t",
        "uint32_t",
        "uint64_t",
        "uintptr_t",
        "intptr_t",
        "ptrdiff_t",
        "bool",
        "FILE",
        "va_list",
        "wchar_t",
        "off_t",
        "time_t",
    }
)

C = LanguageSpec(
    language="c",
    family="c",
    rules=_C_RULES,
    calls=(CallRule(("call_expression",), callee="function", arguments="arguments"),),
    member_access={"field_expression": ("argument", "field")},
    self_names=frozenset(),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=_C_BUILTINS,
    modifier_kinds=frozenset({"storage_class_specifier", "type_qualifier"}),
    block_kinds=frozenset({"compound_statement"}),
    binding_paths=_C_BINDINGS,
)

CPP = LanguageSpec(
    language="cpp",
    family="c",
    rules=(
        Rule(("preproc_include",), Role.IMPORT, name="path", body=None),
        Rule(("preproc_def",), Role.FIELD, body=None, scope="top"),
        Rule(("namespace_definition",), Role.MODULE),
        Rule(
            ("class_specifier",),
            Role.TYPE,
            _CLASS,
            supertypes=("@base_class_clause",),
            require=("body", "field_declaration_list"),
        ),
        Rule(
            ("struct_specifier", "union_specifier"),
            Role.TYPE,
            _STRUCT,
            supertypes=("@base_class_clause",),
            require=("body", "field_declaration_list"),
        ),
        Rule(("enum_specifier",), Role.TYPE, _ENUM, require=("body", "enumerator_list")),
        Rule(("enumerator",), Role.CONSTANT, body=None),
        Rule(("alias_declaration",), Role.TYPE, _TYPE, body=None),
        Rule(
            ("type_definition",),
            Role.TYPE,
            _STRUCT,
            name="declarator",
            body="type/body",
            require=("type/body", "field_declaration_list"),
        ),
        Rule(("type_definition",), Role.TYPE, _TYPE, name="declarator", body=None),
        Rule(
            ("function_definition",),
            Role.CALLABLE,
            name="declarator~function_declarator/declarator*",
            params="declarator~function_declarator/parameters",
            require=("declarator~function_declarator", "function_declarator"),
            return_type="type",
        ),
        Rule(
            ("field_declaration",),
            Role.CALLABLE,
            name="declarator~function_declarator/declarator*",
            params="declarator~function_declarator/parameters",
            body=None,
            require=("declarator~function_declarator", "function_declarator"),
            return_type="type",
        ),
        Rule(("declaration",), Role.SKIP, require=("declarator~function_declarator", "function_declarator")),
        Rule(("declaration",), Role.FIELD, name="declarator*", body=None, scope="member"),
        Rule(("field_declaration",), Role.FIELD, name="declarator*", body=None),
    ),
    calls=(
        CallRule(("call_expression",), callee="function", arguments="arguments"),
        CallRule(("new_expression",), callee="type", arguments="arguments", is_new=True),
    ),
    wrappers={"template_declaration": "@class_specifier|@struct_specifier|@function_definition|@alias_declaration"},
    member_access={"field_expression": ("argument", "field"), "qualified_identifier": ("scope", "name")},
    self_names=frozenset({"this"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=_C_BUILTINS,
    modifier_kinds=frozenset({"storage_class_specifier", "type_qualifier", "virtual_specifier", "virtual"}),
    modifier_words=frozenset({"virtual", "inline", "constexpr", "explicit", "friend", "static", "extern", "mutable"}),
    block_kinds=frozenset({"compound_statement"}),
    binding_paths=_C_BINDINGS,
    default_visibility=Visibility.PUBLIC,
    name_equal_to_type_is_constructor=True,
)

CSHARP = LanguageSpec(
    language="csharp",
    family="csharp",
    rules=(
        Rule(("using_directive",), Role.IMPORT, name="@qualified_name|@identifier", body=None),
        Rule(("file_scoped_namespace_declaration",), Role.PACKAGE, body=None),
        Rule(("namespace_declaration",), Role.MODULE),
        Rule(("class_declaration",), Role.TYPE, _CLASS, supertypes=("@base_list",)),
        Rule(("interface_declaration",), Role.TYPE, _INTERFACE, supertypes=("@base_list",)),
        Rule(("struct_declaration",), Role.TYPE, _STRUCT, supertypes=("@base_list",)),
        Rule(("record_declaration",), Role.TYPE, _RECORD, supertypes=("@base_list",), params="@parameter_list"),
        Rule(("enum_declaration",), Role.TYPE, _ENUM),
        Rule(("enum_member_declaration",), Role.CONSTANT, body=None),
        Rule(("delegate_declaration",), Role.TYPE, _TYPE, body=None),
        Rule(("constructor_declaration",), Role.CALLABLE, params="parameters", is_constructor=True),
        Rule(("destructor_declaration",), Role.CALLABLE, params="parameters"),
        Rule(
            ("method_declaration", "local_function_statement"),
            Role.CALLABLE,
            params="parameters",
            return_type="returns",
        ),
        Rule(("property_declaration", "event_declaration"), Role.FIELD, body=None),
        Rule(
            ("field_declaration", "event_field_declaration"),
            Role.FIELD,
            names="@variable_declaration/@variable_declarator*/name",
            body=None,
        ),
    ),
    calls=(
        CallRule(("invocation_expression",), callee="function", arguments="arguments"),
        CallRule(("object_creation_expression",), callee="type", arguments="arguments", is_new=True),
    ),
    member_access={"member_access_expression": ("expression", "name")},
    self_names=frozenset({"this", "base"}),
    builtin_types=frozenset(
        {
            "var",
            "object",
            "string",
            "int",
            "long",
            "short",
            "byte",
            "bool",
            "double",
            "float",
            "decimal",
            "char",
            "void",
            "dynamic",
            "Task",
            "List",
            "Dictionary",
            "IEnumerable",
            "Func",
            "Action",
        }
    ),
    annotation_kinds=frozenset({"attribute_list"}),
    modifier_kinds=frozenset({"modifier"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameter", "name"),
        ("variable_declarator", "name"),
        ("declaration_expression", "name"),
        ("foreach_statement", "left"),
        ("catch_declaration", "name"),
        ("lambda_expression", "parameters/@identifier*"),
    ),
    default_visibility=Visibility.PRIVATE,
    test_file_patterns=("*Tests.cs", "*Test.cs"),
    name_equal_to_type_is_constructor=True,
)

RUBY = LanguageSpec(
    language="ruby",
    family="ruby",
    rules=(
        Rule(("module",), Role.MODULE),
        Rule(("class",), Role.TYPE, _CLASS, supertypes=("superclass",)),
        Rule(
            ("call",),
            Role.SUPERTYPE,
            names="arguments/@constant*|arguments/@scope_resolution*",
            body=None,
            when=("method", frozenset({"include", "extend", "prepend"})),
        ),
        Rule(
            ("call",),
            Role.FIELD,
            names="arguments/@simple_symbol*",
            body=None,
            when=("method", frozenset({"attr_reader", "attr_writer", "attr_accessor"})),
        ),
        Rule(
            ("call",),
            Role.IMPORT,
            name="arguments/@string",
            body=None,
            when=("method", frozenset({"require", "require_relative", "load"})),
        ),
        Rule(("method", "singleton_method"), Role.CALLABLE, params="parameters"),
        Rule(("assignment",), Role.FIELD, name="left", body=None, scope="member", require=("left", "constant")),
    ),
    calls=(CallRule(("call",), callee="method", arguments="arguments", receiver="receiver"),),
    member_access={"scope_resolution": ("scope", "name")},
    self_names=frozenset({"self"}),
    block_kinds=frozenset({"body_statement", "block_body", "do_block"}),
    binding_paths=(
        ("method_parameters", "@identifier*"),
        ("block_parameters", "@identifier*"),
        ("optional_parameter", "name"),
        ("keyword_parameter", "name"),
        ("splat_parameter", "name"),
        ("hash_splat_parameter", "name"),
        ("block_parameter", "name"),
        ("assignment", "left"),
        ("for", "pattern"),
        ("exception_variable", "@identifier"),
    ),
    constructor_names=frozenset({"initialize"}),
    constructor_call_names=frozenset({"new"}),
    test_file_patterns=("*_spec.rb", "*_test.rb", "test_*.rb"),
    package_separator="::",
    member_separators=frozenset({".", "::", "&."}),
)

PHP = LanguageSpec(
    language="php",
    family="php",
    rules=(
        Rule(("namespace_definition",), Role.MODULE, require=("body", "compound_statement")),
        Rule(("namespace_definition",), Role.PACKAGE, body=None),
        Rule(("namespace_use_declaration",), Role.IMPORT, name=".", names="@namespace_use_clause*", body=None),
        Rule(("class_declaration",), Role.TYPE, _CLASS, supertypes=("@base_clause", "@class_interface_clause")),
        Rule(("interface_declaration",), Role.TYPE, _INTERFACE, supertypes=("@base_clause",)),
        Rule(("trait_declaration",), Role.TYPE, _CLASS),
        Rule(("enum_declaration",), Role.TYPE, _ENUM),
        Rule(("enum_case",), Role.CONSTANT, body=None),
        Rule(("use_declaration",), Role.SUPERTYPE, names="@name*|@qualified_name*", body=None),
        Rule(("method_declaration",), Role.CALLABLE, params="parameters", return_type="return_type"),
        Rule(("function_definition",), Role.CALLABLE, params="parameters", return_type="return_type"),
        Rule(("property_declaration",), Role.FIELD, names="@property_element*/name", body=None),
        Rule(("const_declaration",), Role.FIELD, names="@const_element*/@name", body=None),
    ),
    calls=(
        CallRule(("function_call_expression",), callee="function", arguments="arguments"),
        CallRule(
            ("member_call_expression", "nullsafe_member_call_expression"),
            callee="name",
            arguments="arguments",
            receiver="object",
        ),
        CallRule(("scoped_call_expression",), callee="name", arguments="arguments", receiver="scope"),
        CallRule(("object_creation_expression",), callee="@name|@qualified_name", arguments="@arguments", is_new=True),
    ),
    self_names=frozenset({"this", "self", "static", "parent"}),
    type_ref_kinds=frozenset({"named_type"}),
    builtin_types=frozenset(
        {"int", "string", "float", "bool", "array", "void", "mixed", "null", "callable", "iterable", "object", "never"}
    ),
    annotation_kinds=frozenset({"attribute_list"}),
    modifier_kinds=frozenset(
        {"visibility_modifier", "static_modifier", "abstract_modifier", "final_modifier", "readonly_modifier"}
    ),
    block_kinds=frozenset({"compound_statement"}),
    binding_paths=(
        ("simple_parameter", "name"),
        ("variadic_parameter", "name"),
        ("property_promotion_parameter", "name"),
        ("assignment_expression", "left"),
        ("foreach_statement", "@variable_name*"),
        ("catch_clause", "name"),
    ),
    constructor_names=frozenset({"__construct"}),
    test_file_patterns=("*Test.php",),
    package_separator="\\",
    member_separators=frozenset({"->", "::", "?->"}),
)

KOTLIN = LanguageSpec(
    language="kotlin",
    family="jvm",
    rules=(
        Rule(("package_header",), Role.PACKAGE, name="@qualified_identifier|@identifier", body=None),
        Rule(("import",), Role.IMPORT, name="@qualified_identifier|@identifier", body=None),
        Rule(
            ("class_declaration",),
            Role.TYPE,
            keyword_kinds={
                "class": _CLASS,
                "interface": _INTERFACE,
                "enum": _ENUM,
                "annotation": SymbolKind.ANNOTATION,
            },
            body="@class_body|@enum_class_body",
            supertypes=("@delegation_specifiers",),
            params="@primary_constructor/@class_parameters",
            members=("@primary_constructor/@class_parameters",),
        ),
        Rule(
            ("object_declaration",),
            Role.TYPE,
            _CLASS,
            name="name|@identifier",
            body="@class_body",
            supertypes=("@delegation_specifiers",),
        ),
        Rule(
            ("companion_object",), Role.TYPE, _CLASS, name="@identifier", default_name="Companion", body="@class_body"
        ),
        Rule(("type_alias",), Role.TYPE, _TYPE, name="name|@identifier", body=None),
        Rule(("enum_entry",), Role.CONSTANT, name="name|@identifier", body=None),
        Rule(
            ("function_declaration",),
            Role.CALLABLE,
            name="name|@identifier",
            params="@function_value_parameters",
            body="@function_body",
            return_type="@user_type|@nullable_type",
        ),
        Rule(
            ("secondary_constructor",),
            Role.CALLABLE,
            name=None,
            default_name="constructor",
            params="@function_value_parameters",
            body="@block",
            is_constructor=True,
        ),
        Rule(("class_parameter",), Role.FIELD, name="@identifier", body=None, keywords=frozenset({"val", "var"})),
        Rule(
            ("property_declaration",), Role.FIELD, name="@variable_declaration/@identifier", body=None, scope="member"
        ),
    ),
    calls=(CallRule(("call_expression",), callee="#0", arguments="@value_arguments"),),
    member_access={"navigation_expression": ("#0", "#-1/@identifier|#-1")},
    self_names=frozenset({"this", "this_expression"}),
    type_ref_kinds=frozenset({"user_type"}),
    builtin_types=frozenset(
        {
            "Int",
            "Long",
            "Short",
            "Byte",
            "Float",
            "Double",
            "Boolean",
            "Char",
            "String",
            "Unit",
            "Any",
            "Nothing",
            "List",
            "MutableList",
            "Map",
            "MutableMap",
            "Set",
            "Array",
            "Pair",
        }
    ),
    annotation_kinds=frozenset({"annotation"}),
    modifier_kinds=frozenset({"modifiers"}),
    modifier_words=frozenset({"val", "var", "suspend", "inline", "operator", "infix", "data", "open", "abstract"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameter", "@identifier"),
        ("class_parameter", "@identifier"),
        ("variable_declaration", "@identifier"),
        ("lambda_parameters", "**identifier"),
        ("for_statement", "@variable_declaration/@identifier"),
    ),
    capitalized_call_is_new=True,
    test_file_patterns=("*Test.kt", "*Tests.kt", "*Spec.kt"),
)

SWIFT = LanguageSpec(
    language="swift",
    family="swift",
    rules=(
        Rule(("import_declaration",), Role.IMPORT, name="@identifier", body=None),
        Rule(("class_declaration",), Role.EXTENSION, owner="name", keywords=frozenset({"extension"})),
        Rule(
            ("class_declaration",),
            Role.TYPE,
            keyword_kinds={"class": _CLASS, "struct": _STRUCT, "enum": _ENUM, "actor": _CLASS, "indirect": _ENUM},
            supertypes=("@inheritance_specifier*/inherits_from",),
        ),
        Rule(("protocol_declaration",), Role.TYPE, _INTERFACE, supertypes=("@inheritance_specifier*/inherits_from",)),
        Rule(("typealias_declaration",), Role.TYPE, _TYPE, body=None),
        Rule(("enum_entry",), Role.CONSTANT, names="name+", body=None),
        Rule(("function_declaration",), Role.CALLABLE, params=".", return_type="@user_type|@optional_type"),
        Rule(("protocol_function_declaration",), Role.CALLABLE, params=".", body=None),
        Rule(("init_declaration",), Role.CALLABLE, name=None, default_name="init", params=".", is_constructor=True),
        Rule(("deinit_declaration",), Role.CALLABLE, name=None, default_name="deinit", params=None),
        Rule(
            ("property_declaration",),
            Role.FIELD,
            names="@pattern*/bound_identifier|name/bound_identifier",
            body=None,
            scope="member",
        ),
        Rule(("protocol_property_declaration",), Role.FIELD, name="name/bound_identifier", body=None),
    ),
    calls=(CallRule(("call_expression",), callee="#0", arguments="@call_suffix/@value_arguments"),),
    member_access={"navigation_expression": ("target", "suffix/suffix")},
    self_names=frozenset({"self", "Self", "self_expression", "super"}),
    parameter_kinds=frozenset({"parameter"}),
    type_ref_kinds=frozenset({"user_type"}),
    builtin_types=frozenset(
        {
            "Int",
            "Double",
            "Float",
            "Bool",
            "String",
            "Character",
            "Void",
            "Any",
            "AnyObject",
            "Array",
            "Dictionary",
            "Set",
            "Optional",
            "Self",
        }
    ),
    annotation_kinds=frozenset({"attribute"}),
    modifier_kinds=frozenset({"modifiers"}),
    modifier_words=frozenset(
        {"static", "override", "final", "mutating", "convenience", "required", "lazy", "weak", "unowned"}
    ),
    block_kinds=frozenset({"statements"}),
    binding_paths=(
        ("parameter", "name"),
        ("property_declaration", "name/bound_identifier"),
        ("lambda_parameter", "name"),
    ),
    capitalized_call_is_new=True,
    default_visibility=Visibility.PACKAGE,
    test_file_patterns=("*Tests.swift", "*Test.swift"),
)

SCALA = LanguageSpec(
    language="scala",
    family="jvm",
    rules=(
        Rule(("package_clause",), Role.PACKAGE, body=None),
        Rule(("import_declaration",), Role.IMPORT, name=".", body=None),
        Rule(
            ("class_definition",),
            Role.TYPE,
            _CLASS,
            supertypes=("extend",),
            params="class_parameters",
            members=("class_parameters",),
        ),
        Rule(("trait_definition",), Role.TYPE, _INTERFACE, supertypes=("extend",)),
        Rule(("object_definition",), Role.TYPE, _CLASS, supertypes=("extend",)),
        Rule(("enum_definition",), Role.TYPE, _ENUM, supertypes=("extend",)),
        Rule(("type_definition",), Role.TYPE, _TYPE, body=None),
        Rule(("enum_case_definitions",), Role.CONSTANT, names="@simple_enum_case*/name", body=None),
        Rule(("function_definition",), Role.CALLABLE, params="parameters", return_type="return_type"),
        Rule(("function_declaration",), Role.CALLABLE, params="parameters", body=None, return_type="return_type"),
        Rule(("class_parameter",), Role.FIELD, body=None, keywords=frozenset({"val", "var"})),
        Rule(("val_definition", "var_definition"), Role.FIELD, name="pattern", body=None, scope="member"),
        Rule(("val_declaration", "var_declaration"), Role.FIELD, body=None),
    ),
    calls=(
        CallRule(("call_expression",), callee="function", arguments="arguments"),
        CallRule(("instance_expression",), callee="#0", arguments="#0/arguments", is_new=True),
    ),
    member_access={"field_expression": ("value", "field")},
    self_names=frozenset({"this", "super"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=frozenset(
        {
            "Int",
            "Long",
            "Double",
            "Float",
            "Boolean",
            "String",
            "Unit",
            "Any",
            "AnyRef",
            "Nothing",
            "Option",
            "List",
            "Seq",
            "Map",
            "Set",
        }
    ),
    annotation_kinds=frozenset({"annotation"}),
    modifier_kinds=frozenset({"modifiers"}),
    modifier_words=frozenset({"val", "var", "case", "implicit", "lazy", "override", "sealed", "abstract", "final"}),
    block_kinds=frozenset({"block", "template_body"}),
    binding_paths=(
        ("parameter", "name"),
        ("class_parameter", "name"),
        ("val_definition", "pattern"),
        ("var_definition", "pattern"),
        ("lambda_expression", "parameters/@identifier*"),
        ("binding", "name"),
    ),
    capitalized_call_is_new=True,
    test_file_patterns=("*Spec.scala", "*Test.scala", "*Suite.scala"),
)

DART = LanguageSpec(
    language="dart",
    family="dart",
    rules=(
        Rule(("import_or_export",), Role.IMPORT, name="**uri", body=None),
        Rule(("class_definition",), Role.TYPE, _CLASS, supertypes=("superclass", "interfaces", "@mixins")),
        Rule(("mixin_declaration",), Role.TYPE, _CLASS, name="@identifier", body="@class_body"),
        Rule(
            ("extension_declaration",),
            Role.EXTENSION,
            owner="@type_identifier|**type_identifier",
            body="@extension_body",
        ),
        Rule(("enum_declaration",), Role.TYPE, _ENUM),
        Rule(("enum_constant",), Role.CONSTANT, body=None),
        Rule(
            ("method_signature",),
            Role.CALLABLE,
            name="@function_signature/name|@getter_signature/name|@setter_signature/name|@constructor_signature/name",
            params="@function_signature/@formal_parameter_list|@constructor_signature/parameters",
            body=">",
            body_kind="function_body",
        ),
        Rule(
            ("function_signature",),
            Role.CALLABLE,
            params="@formal_parameter_list",
            body=">",
            body_kind="function_body",
        ),
        Rule(
            ("declaration",),
            Role.CALLABLE,
            name="@constructor_signature/name",
            params="@constructor_signature/parameters",
            body="@function_body",
            require=("@constructor_signature", "constructor_signature"),
            is_constructor=True,
        ),
        Rule(
            ("declaration",),
            Role.FIELD,
            names="@initialized_identifier_list/@initialized_identifier*/@identifier|@static_final_declaration_list/@static_final_declaration*/@identifier",
            body=None,
        ),
        Rule(
            ("initialized_variable_definition",),
            Role.FIELD,
            name="name",
            body=None,
            scope="member",
        ),
    ),
    calls=(
        CallRule(
            ("selector",),
            callee="<",
            arguments="@argument_part/@arguments",
            require=("@argument_part", "argument_part"),
        ),
    ),
    member_access={"selector": ("<", "@unconditional_assignable_selector/@identifier")},
    self_names=frozenset({"this", "super"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=frozenset(
        {
            "int",
            "double",
            "num",
            "String",
            "bool",
            "void",
            "dynamic",
            "List",
            "Map",
            "Set",
            "Future",
            "Stream",
            "Object",
            "Null",
            "Iterable",
        }
    ),
    annotation_kinds=frozenset({"annotation"}),
    modifier_words=frozenset({"static", "final", "const", "abstract", "late", "external", "factory", "async"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("formal_parameter", "name"),
        ("initialized_variable_definition", "name"),
        ("initialized_identifier", "@identifier"),
        ("constructor_param", "@identifier"),
    ),
    capitalized_call_is_new=True,
    underscore_is_private=True,
    test_file_patterns=("*_test.dart",),
    name_equal_to_type_is_constructor=True,
)

LUA = LanguageSpec(
    language="lua",
    family="lua",
    rules=(
        Rule(("function_declaration",), Role.CALLABLE, params="parameters"),
        Rule(
            ("variable_declaration",),
            Role.FIELD,
            names="@assignment_statement/@variable_list/@identifier*",
            body=None,
            scope="member",
        ),
        Rule(
            ("assignment_statement",),
            Role.FIELD,
            names="@variable_list/@identifier*",
            body=None,
            scope="top",
        ),
    ),
    calls=(CallRule(("function_call",), callee="name", arguments="arguments"),),
    wrappers={"variable_declaration": "@function_declaration"},
    member_access={"dot_index_expression": ("table", "field"), "method_index_expression": ("table", "method")},
    self_names=frozenset({"self"}),
    modifier_words=frozenset({"local"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameters", "@identifier*"),
        ("variable_declaration", "@assignment_statement/@variable_list/@identifier*"),
        ("for_generic_clause", "@variable_list/@identifier*"),
        ("for_numeric_clause", "name"),
    ),
    call_exclusions=frozenset({"require", "pcall", "print", "type", "pairs", "ipairs", "tostring", "tonumber"}),
    test_file_patterns=("*_spec.lua", "*_test.lua"),
    member_separators=frozenset({".", ":"}),
)

ZIG = LanguageSpec(
    language="zig",
    family="zig",
    rules=(
        Rule(
            ("variable_declaration",),
            Role.IMPORT,
            name="@builtin_function/@arguments/@string",
            body=None,
            when=("@builtin_function/@builtin_identifier", frozenset({"import"})),
        ),
        Rule(
            ("variable_declaration",),
            Role.TYPE,
            _STRUCT,
            name="@identifier",
            body="@struct_declaration",
            require=("@struct_declaration", "struct_declaration"),
        ),
        Rule(
            ("variable_declaration",),
            Role.TYPE,
            _ENUM,
            name="@identifier",
            body="@enum_declaration",
            require=("@enum_declaration", "enum_declaration"),
        ),
        Rule(
            ("variable_declaration",),
            Role.TYPE,
            _STRUCT,
            name="@identifier",
            body="@union_declaration",
            require=("@union_declaration", "union_declaration"),
        ),
        Rule(
            ("variable_declaration",),
            Role.TYPE,
            _TYPE,
            name="@identifier",
            body=None,
            require=("@opaque_declaration", "opaque_declaration"),
        ),
        Rule(("function_declaration",), Role.CALLABLE, params="@parameters", return_type="type"),
        Rule(
            ("test_declaration",),
            Role.CALLABLE,
            name="@string/@string_content|@identifier",
            default_name="test",
            body="@block",
        ),
        Rule(("container_field",), Role.CONSTANT, body=None, require=("..", "enum_declaration")),
        Rule(("container_field",), Role.FIELD, body=None),
        Rule(("variable_declaration",), Role.FIELD, name="@identifier", body=None, scope="member"),
    ),
    calls=(
        CallRule(("call_expression",), callee="function", arguments=".", arguments_offset=1),
        CallRule(("builtin_function",), callee="@builtin_identifier", arguments="@arguments"),
    ),
    member_access={"field_expression": ("#0", "#-1")},
    self_names=frozenset({"self", "Self"}),
    modifier_words=frozenset({"pub", "export", "extern", "inline", "noinline", "threadlocal", "const", "var"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(
        ("parameter", "name"),
        ("variable_declaration", "@identifier"),
        ("payload", "@identifier*"),
    ),
    call_exclusions=frozenset({"import"}),
    default_visibility=Visibility.PRIVATE,
    member_separators=frozenset({"."}),
)


_ELIXIR_DEFINERS = frozenset({"def", "defp", "defmacro", "defmacrop", "defguard", "defguardp", "defdelegate"})
_ELIXIR_SYNTAX = frozenset(
    {
        "def",
        "defp",
        "defmacro",
        "defmacrop",
        "defguard",
        "defguardp",
        "defmodule",
        "defstruct",
        "defdelegate",
        "defimpl",
        "defprotocol",
        "defexception",
        "defoverridable",
        "import",
        "alias",
        "use",
        "require",
        "moduledoc",
        "doc",
        "spec",
        "type",
        "typep",
        "impl",
        "behaviour",
        "callback",
        "if",
        "unless",
        "case",
        "cond",
        "with",
        "for",
        "fn",
        "raise",
        "quote",
        "unquote",
        "receive",
        "try",
    }
)

ELIXIR = LanguageSpec(
    language="elixir",
    family="elixir",
    rules=(
        Rule(
            ("call",),
            Role.MODULE,
            name="@arguments/@alias",
            body="@do_block",
            when=("target", frozenset({"defmodule", "defprotocol", "defimpl"})),
        ),
        Rule(
            ("call",),
            Role.IMPORT,
            name="@arguments/@alias",
            body=None,
            when=("target", frozenset({"import", "alias", "use", "require"})),
        ),
        Rule(
            ("call",),
            Role.CALLABLE,
            name="@arguments/@call/target|@arguments/@binary_operator/@call/target|@arguments/@identifier",
            params="@arguments/@call/@arguments|@arguments/@binary_operator/@call/@arguments",
            body="@do_block|@arguments/@keywords",
            when=("target", _ELIXIR_DEFINERS),
        ),
    ),
    calls=(CallRule(("call",), callee="target", arguments="@arguments"),),
    member_access={"dot": ("left", "right")},
    self_names=frozenset({"__MODULE__"}),
    block_kinds=frozenset({"do_block"}),
    binding_paths=(("binary_operator", "left"), ("arguments", "@identifier*")),
    call_exclusions=_ELIXIR_SYNTAX,
    transparent_kinds=frozenset({"arguments"}),
    private_words=frozenset({"defp", "defmacrop", "defguardp"}),
    test_file_patterns=("*_test.exs",),
)

ERLANG = LanguageSpec(
    language="erlang",
    family="erlang",
    rules=(
        Rule(("module_attribute",), Role.PACKAGE, body=None),
        Rule(("record_decl",), Role.TYPE, _RECORD, body="."),
        Rule(("record_field",), Role.FIELD, body=None),
        Rule(("type_alias", "opaque"), Role.TYPE, _TYPE, name="name/name|**atom", body=None),
        Rule(("fun_decl",), Role.CALLABLE, name="clause/name", params="clause/args", body="clause/body"),
    ),
    calls=(CallRule(("call",), callee="expr", arguments="args"),),
    member_access={"remote": ("module/module", "fun")},
    block_kinds=frozenset({"clause_body"}),
    binding_paths=(("expr_args", "@var*"), ("match_expr", "lhs")),
    test_file_patterns=("*_SUITE.erl", "*_tests.erl"),
)

HASKELL = LanguageSpec(
    language="haskell",
    family="haskell",
    rules=(
        Rule(("header",), Role.PACKAGE, name="module", body=None),
        Rule(("import",), Role.IMPORT, name="module", body=None),
        Rule(("data_type",), Role.TYPE, _TYPE, body="constructors"),
        Rule(
            ("data_constructor",), Role.CONSTANT, name="constructor/@constructor|constructor/**constructor", body=None
        ),
        Rule(("newtype",), Role.TYPE, _TYPE, body=None),
        Rule(("type_synomym",), Role.TYPE, _TYPE, body=None),
        Rule(("class",), Role.TYPE, _INTERFACE, body="declarations"),
        Rule(("instance",), Role.EXTENSION, owner="patterns/@name|patterns/**name", trait="name", body="declarations"),
        Rule(("signature",), Role.SKIP),
        Rule(("function",), Role.CALLABLE, params="patterns", body="match"),
        Rule(("bind",), Role.FIELD, body=None, scope="member"),
    ),
    calls=(CallRule(("apply",), callee="function*", arguments=None),),
    block_kinds=frozenset(),
    binding_paths=(("function", "patterns/@variable*"), ("bind", "name")),
    test_file_patterns=("*Spec.hs", "*Test.hs"),
)

OCAML = LanguageSpec(
    language="ocaml",
    family="ocaml",
    rules=(
        Rule(("open_module",), Role.IMPORT, name="@module_path", body=None),
        Rule(("module_binding",), Role.MODULE, name="@module_name"),
        Rule(("type_binding",), Role.TYPE, _TYPE),
        Rule(("constructor_declaration",), Role.CONSTANT, name="@constructor_name", body=None),
        Rule(("field_declaration",), Role.FIELD, name="@field_name", body=None),
        Rule(("class_binding",), Role.TYPE, _CLASS, name="@class_name"),
        Rule(("method_definition",), Role.CALLABLE, name="@method_name"),
        Rule(("let_binding",), Role.CALLABLE, name="pattern", params=".", require=("@parameter", "parameter")),
        Rule(("let_binding",), Role.FIELD, name="pattern", body=None, scope="member"),
    ),
    calls=(CallRule(("application_expression",), callee="function", arguments=".", arguments_offset=1),),
    member_access={"value_path": ("@module_path", "@value_name"), "field_get_expression": ("record", "field")},
    parameter_kinds=frozenset({"parameter"}),
    block_kinds=frozenset({"sequence_expression"}),
    binding_paths=(("parameter", "pattern"), ("let_binding", "pattern")),
    test_file_patterns=("test_*.ml", "*_test.ml"),
)

JULIA = LanguageSpec(
    language="julia",
    family="julia",
    rules=(
        Rule(("using_statement", "import_statement"), Role.IMPORT, name="#0", body=None),
        Rule(("module_definition",), Role.MODULE, body="@block"),
        Rule(
            ("struct_definition",),
            Role.TYPE,
            _STRUCT,
            name="@type_head/@identifier|@type_head/**identifier",
            body="@block",
        ),
        Rule(
            ("abstract_definition",), Role.TYPE, _TYPE, name="@type_head/@identifier|@type_head/**identifier", body=None
        ),
        Rule(
            ("function_definition", "macro_definition"),
            Role.CALLABLE,
            name="@signature/@call_expression/#0|@signature/**identifier|@identifier",
            params="@signature/@call_expression/@argument_list",
            body="@block",
        ),
        Rule(
            ("assignment",),
            Role.CALLABLE,
            name="#0/#0",
            params="#0/@argument_list",
            body="#-1",
            require=("#0", "call_expression"),
        ),
        Rule(("const_statement",), Role.FIELD, name="@assignment/#0|**identifier", body=None),
        Rule(("typed_expression",), Role.FIELD, name="#0", body=None, scope="type"),
    ),
    calls=(CallRule(("call_expression",), callee="#0", arguments="@argument_list"),),
    member_access={"field_expression": ("#0", "#-1")},
    block_kinds=frozenset({"block"}),
    binding_paths=(("argument_list", "@identifier*"), ("assignment", "#0"), ("typed_expression", "#0")),
    test_file_patterns=("test_*.jl", "*_test.jl", "runtests.jl"),
)

R = LanguageSpec(
    language="r",
    family="r",
    rules=(
        Rule(
            ("call",),
            Role.IMPORT,
            name="arguments/@argument/value",
            body=None,
            when=("function", frozenset({"library", "require", "source"})),
        ),
        Rule(
            ("binary_operator",),
            Role.CALLABLE,
            name="lhs",
            params="rhs/parameters",
            body="rhs/body",
            require=("rhs", "function_definition"),
        ),
        Rule(("binary_operator",), Role.FIELD, name="lhs", body=None, scope="top", require=("lhs", "identifier")),
    ),
    calls=(CallRule(("call",), callee="function", arguments="arguments"),),
    member_access={"namespace_operator": ("lhs", "rhs"), "extract_operator": ("lhs", "rhs")},
    parameter_kinds=frozenset({"parameter"}),
    block_kinds=frozenset({"braced_expression"}),
    binding_paths=(("parameter", "name"), ("binary_operator", "lhs"), ("for_statement", "variable")),
    test_file_patterns=("test-*.R", "test_*.R", "test-*.r"),
)

PERL = LanguageSpec(
    language="perl",
    family="perl",
    rules=(
        Rule(("package_statement",), Role.PACKAGE, name="@package_name", body=None),
        Rule(("use_no_statement",), Role.IMPORT, name="package_name|@package_name", body=None),
        Rule(("function_definition",), Role.CALLABLE),
    ),
    calls=(
        CallRule(
            ("call_expression_with_args_with_brackets", "call_expression_with_spaced_args"),
            callee="@call_expression_with_bareword/function_name",
            arguments="args",
        ),
        CallRule(("call_expression_with_bareword",), callee="function_name", arguments=None),
        CallRule(
            ("method_invocation",),
            callee="function_name",
            arguments="args/@arguments",
            receiver="object_return_value|package_name",
        ),
    ),
    self_names=frozenset({"self", "$self"}),
    block_kinds=frozenset({"block"}),
    binding_paths=(("variable_declaration", "**scalar_variable"), ("variable_declaration", "**array_variable")),
    call_exclusions=frozenset({"shift", "push", "pop", "print", "return", "die", "bless", "defined", "scalar", "ref"}),
    test_file_patterns=("*.t",),
    package_separator="::",
    member_separators=frozenset({"->", "::"}),
)

BASH = LanguageSpec(
    language="bash",
    family="bash",
    rules=(
        Rule(("command",), Role.IMPORT, name="argument", body=None, when=("name", frozenset({"source", "."}))),
        Rule(("function_definition",), Role.CALLABLE),
        Rule(("variable_assignment",), Role.FIELD, body=None, scope="top"),
    ),
    calls=(CallRule(("command",), callee="name", arguments="argument+"),),
    callee_kinds=frozenset({"command_name"}),
    block_kinds=frozenset({"compound_statement"}),
    binding_paths=(("variable_assignment", "name"), ("for_statement", "variable")),
    call_exclusions=frozenset(
        {"echo", "cd", "exit", "return", "export", "local", "set", "shift", "test", "printf", "eval", "exec"}
    ),
)

POWERSHELL = LanguageSpec(
    language="powershell",
    family="powershell",
    rules=(
        Rule(("class_statement",), Role.TYPE, _CLASS, name="@simple_name", body="."),
        Rule(("enum_statement",), Role.TYPE, _ENUM, name="@simple_name", body="."),
        Rule(
            ("function_statement",),
            Role.CALLABLE,
            name="@function_name",
            params="@function_parameter_declaration/@parameter_list|**param_block/@parameter_list",
            body="@script_block",
        ),
        Rule(
            ("class_method_definition",),
            Role.CALLABLE,
            name="@simple_name",
            params="@class_method_parameter_list",
            body="@script_block",
        ),
        Rule(("class_property_definition",), Role.FIELD, name="@variable", body=None),
    ),
    calls=(CallRule(("command",), callee="@command_name", arguments=None),),
    callee_kinds=frozenset({"command_name"}),
    self_names=frozenset({"this"}),
    block_kinds=frozenset({"statement_list"}),
    binding_paths=(("script_parameter", "@variable"), ("class_method_parameter", "@variable")),
    test_file_patterns=("*.Tests.ps1",),
    name_equal_to_type_is_constructor=True,
)

SOLIDITY = LanguageSpec(
    language="solidity",
    family="solidity",
    rules=(
        Rule(("import_directive",), Role.IMPORT, name="source", body=None),
        Rule(
            ("contract_declaration", "library_declaration"),
            Role.TYPE,
            _CLASS,
            supertypes=("@inheritance_specifier*/ancestor",),
        ),
        Rule(("interface_declaration",), Role.TYPE, _INTERFACE, supertypes=("@inheritance_specifier*/ancestor",)),
        Rule(("struct_declaration",), Role.TYPE, _STRUCT),
        Rule(("enum_declaration",), Role.TYPE, _ENUM),
        Rule(("enum_value",), Role.CONSTANT, name=".", body=None),
        Rule(("struct_member", "state_variable_declaration"), Role.FIELD, body=None),
        Rule(
            ("constructor_definition",),
            Role.CALLABLE,
            name=None,
            default_name="constructor",
            params=".",
            is_constructor=True,
        ),
        Rule(("function_definition", "modifier_definition"), Role.CALLABLE, params=".", return_type="return_type"),
        Rule(("event_definition", "error_declaration"), Role.CALLABLE, params=".", body=None),
        Rule(("user_defined_type_definition",), Role.TYPE, _TYPE, body=None),
    ),
    calls=(
        CallRule(("call_expression",), callee="function|#0", arguments="@call_argument*"),
        CallRule(("new_expression",), callee="name|@type_name", arguments=None, is_new=True),
        CallRule(("emit_statement",), callee="name|#0", arguments="@call_argument*"),
    ),
    member_access={"member_expression": ("object", "property")},
    self_names=frozenset({"this", "super"}),
    parameter_kinds=frozenset({"parameter", "event_parameter", "error_parameter"}),
    transparent_kinds=frozenset({"expression"}),
    type_ref_kinds=frozenset({"user_defined_type"}),
    builtin_types=frozenset(
        {
            "uint",
            "int",
            "address",
            "bool",
            "string",
            "bytes",
            "mapping",
            "bytes32",
            "uint256",
            "int256",
            "uint8",
            "payable",
        }
    ),
    modifier_kinds=frozenset({"visibility", "state_mutability", "override_specifier", "virtual"}),
    modifier_words=frozenset({"constant", "immutable", "payable", "indexed", "anonymous"}),
    block_kinds=frozenset({"function_body", "block_statement"}),
    binding_paths=(("parameter", "name"), ("variable_declaration", "name")),
    test_file_patterns=("*.t.sol", "*Test.sol"),
)

GROOVY = LanguageSpec(
    language="groovy",
    family="jvm",
    rules=(
        Rule(("groovy_package",), Role.PACKAGE, name="@qualified_name", body=None),
        Rule(("groovy_import",), Role.IMPORT, name="import", body=None),
        Rule(("class_definition",), Role.TYPE, _CLASS, supertypes=("superclass", "@ERROR")),
        Rule(
            ("function_definition", "function_declaration"),
            Role.CALLABLE,
            name="function",
            params="parameters",
            return_type="type",
        ),
        Rule(("declaration",), Role.FIELD, body=None, scope="member"),
    ),
    calls=(CallRule(("function_call",), callee="function", arguments="args"),),
    member_access={"dotted_identifier": ("#0", "#-1")},
    self_names=frozenset({"this", "super"}),
    annotation_kinds=frozenset({"annotation"}),
    modifier_kinds=frozenset({"modifier", "access_modifier"}),
    capitalized_call_is_new=True,
    block_kinds=frozenset({"closure"}),
    binding_paths=(("parameter", "name"), ("declaration", "name")),
    test_file_patterns=("*Test.groovy", "*Spec.groovy"),
)

OBJC = LanguageSpec(
    language="objc",
    family="c",
    rules=(
        Rule(("preproc_include", "module_import"), Role.IMPORT, name="path|#0", body=None),
        Rule(
            ("class_interface",),
            Role.TYPE,
            _CLASS,
            name="@identifier",
            supertypes=("superclass", "@parameterized_arguments"),
            body=".",
        ),
        Rule(("class_implementation",), Role.EXTENSION, owner="@identifier", body="."),
        Rule(("protocol_declaration",), Role.TYPE, _INTERFACE, name="@identifier", body="."),
        Rule(("method_declaration",), Role.CALLABLE, name="@identifier", params=".", body=None),
        Rule(("method_definition",), Role.CALLABLE, name="@identifier", params=".", body="@compound_statement"),
        Rule(("property_declaration",), Role.FIELD, name="**identifier", body=None),
        Rule(("instance_variable",), Role.FIELD, name="**identifier", body=None),
        *_C_RULES,
    ),
    calls=(
        CallRule(("call_expression",), callee="function", arguments="arguments"),
        CallRule(("message_expression",), callee="method", arguments=None, receiver="receiver"),
    ),
    wrappers={"implementation_definition": "@method_definition|@function_definition"},
    member_access={"field_expression": ("argument", "field")},
    self_names=frozenset({"self", "super"}),
    parameter_kinds=frozenset({"method_parameter", "parameter_declaration"}),
    type_ref_kinds=frozenset({"type_identifier"}),
    builtin_types=_C_BUILTINS
    | frozenset({"id", "instancetype", "NSString", "NSObject", "NSArray", "NSDictionary", "BOOL"}),
    modifier_kinds=frozenset({"storage_class_specifier", "type_qualifier"}),
    block_kinds=frozenset({"compound_statement"}),
    binding_paths=_C_BINDINGS,
)

_LISP_SPECIAL = frozenset(
    {
        "define",
        "define-syntax",
        "define-record-type",
        "define-struct",
        "struct",
        "lambda",
        "let",
        "let*",
        "letrec",
        "if",
        "cond",
        "case",
        "when",
        "unless",
        "do",
        "begin",
        "set!",
        "quote",
        "quasiquote",
        "unquote",
        "and",
        "or",
        "not",
        "require",
        "provide",
        "import",
        "export",
        "module",
        "defn",
        "defn-",
        "def",
        "defonce",
        "defmacro",
        "defmulti",
        "defmethod",
        "defrecord",
        "deftype",
        "defprotocol",
        "definterface",
        "ns",
        "fn",
        "loop",
        "recur",
        "->",
        "->>",
        "doto",
        "try",
        "catch",
        "finally",
        "throw",
        "for",
        "doseq",
        "dotimes",
        "while",
        "letfn",
        "binding",
        "with-open",
        "cond->",
        "some->",
        "if-let",
        "when-let",
        "if-not",
        "when-not",
        "comment",
        "declare",
        "extend-protocol",
        "extend-type",
        "reify",
        "proxy",
    }
)

CLOJURE = LanguageSpec(
    language="clojure",
    family="clojure",
    rules=(
        Rule(("list_lit",), Role.PACKAGE, name="#1/name", body=None, when=("#0/name", frozenset({"ns"}))),
        Rule(
            ("list_lit",),
            Role.CALLABLE,
            name="#1/name",
            params="#2",
            body=".",
            when=("#0/name", frozenset({"defn", "defn-", "defmacro", "defmulti", "defmethod"})),
        ),
        Rule(("list_lit",), Role.FIELD, name="#1/name", body=".", when=("#0/name", frozenset({"def", "defonce"}))),
        Rule(
            ("list_lit",),
            Role.TYPE,
            _RECORD,
            name="#1/name",
            body=".",
            when=("#0/name", frozenset({"defrecord", "deftype"})),
        ),
        Rule(
            ("list_lit",),
            Role.TYPE,
            _INTERFACE,
            name="#1/name",
            body=".",
            when=("#0/name", frozenset({"defprotocol", "definterface"})),
        ),
        Rule(
            ("list_lit",),
            Role.EXTENSION,
            owner="#1/name",
            body=".",
            when=("#0/name", frozenset({"extend-protocol", "extend-type"})),
        ),
    ),
    calls=(CallRule(("list_lit",), callee="#0/name", arguments=".", arguments_offset=1),),
    parameter_kinds=frozenset({"sym_lit", "vec_lit", "map_lit"}),
    block_kinds=frozenset(),
    binding_paths=(("vec_lit", "@sym_lit*/name"),),
    call_exclusions=_LISP_SPECIAL,
    symbol_namespace_separator="/",
    private_words=frozenset({"defn-"}),
    test_file_patterns=("*_test.clj", "*_test.cljs", "*_test.cljc"),
)

_SCHEME_RULES = (
    Rule(
        ("list",),
        Role.CALLABLE,
        name="#1/#0",
        params="#1",
        params_offset=1,
        body=".",
        when=("#0", frozenset({"define", "define-syntax", "define-public"})),
        require=("#1", "list"),
    ),
    Rule(
        ("list",),
        Role.FIELD,
        name="#1",
        body=".",
        when=("#0", frozenset({"define", "define-public"})),
        require=("#1", "symbol"),
    ),
    Rule(
        ("list",),
        Role.TYPE,
        _STRUCT,
        name="#1",
        body="#2",
        when=("#0", frozenset({"struct", "define-struct", "define-record-type"})),
    ),
    Rule(("list",), Role.IMPORT, name="#1", body=None, when=("#0", frozenset({"require", "import", "load"}))),
)

SCHEME = LanguageSpec(
    language="scheme",
    family="scheme",
    rules=_SCHEME_RULES,
    calls=(CallRule(("list",), callee="#0", arguments=".", arguments_offset=1),),
    block_kinds=frozenset(),
    call_exclusions=_LISP_SPECIAL,
    test_file_patterns=("*-test.scm", "test-*.scm"),
)

RACKET = LanguageSpec(
    language="racket",
    family="scheme",
    rules=_SCHEME_RULES,
    calls=(CallRule(("list",), callee="#0", arguments=".", arguments_offset=1),),
    block_kinds=frozenset(),
    call_exclusions=_LISP_SPECIAL,
    test_file_patterns=("*-test.rkt", "test-*.rkt"),
)

NIX = LanguageSpec(
    language="nix",
    family="nix",
    rules=(
        Rule(
            ("binding",),
            Role.CALLABLE,
            name="attrpath",
            params="expression/formals|expression",
            body="expression/body",
            require=("expression", "function_expression"),
        ),
        Rule(("binding",), Role.FIELD, name="attrpath", body="expression"),
    ),
    calls=(CallRule(("apply_expression",), callee="function*", arguments=None),),
    member_access={"select_expression": ("expression", "attrpath")},
    parameter_kinds=frozenset({"formal", "identifier"}),
    block_kinds=frozenset(),
    binding_paths=(("formal", "name"), ("function_expression", "universal")),
    call_exclusions=frozenset({"import", "map", "toString", "throw", "abort", "inherit"}),
)

FORTRAN = LanguageSpec(
    language="fortran",
    family="fortran",
    rules=(
        Rule(("use_statement",), Role.IMPORT, name="@module_name", body=None),
        Rule(("module",), Role.MODULE, name="@module_statement/@name", body="."),
        Rule(("program",), Role.MODULE, name="@program_statement/@name", body="."),
        Rule(
            ("derived_type_definition",),
            Role.TYPE,
            _STRUCT,
            name="@derived_type_statement/**type_name|@derived_type_statement/@type_name",
            body=".",
        ),
        Rule(
            ("function",),
            Role.CALLABLE,
            name="@function_statement/name",
            params="@function_statement/parameters",
            body=".",
        ),
        Rule(
            ("subroutine",),
            Role.CALLABLE,
            name="@subroutine_statement/name",
            params="@subroutine_statement/parameters",
            body=".",
        ),
        Rule(("variable_declaration",), Role.FIELD, names="declarator+", body=None, scope="member"),
    ),
    calls=(
        CallRule(("call_expression",), callee="#0", arguments="@argument_list"),
        CallRule(("subroutine_call",), callee="subroutine", arguments="@argument_list"),
    ),
    block_kinds=frozenset(),
    binding_paths=(("variable_declaration", "declarator+"),),
    call_exclusions=frozenset(
        {"size", "len", "abs", "max", "min", "sqrt", "print", "write", "read", "allocate", "trim"}
    ),
)

SQL = LanguageSpec(
    language="sql",
    family="sql",
    rules=(
        Rule(
            ("create_function",),
            Role.CALLABLE,
            name="@object_reference/name",
            params="@function_arguments",
            body="@function_body",
        ),
        Rule(("create_table",), Role.TYPE, _TYPE, name="@object_reference/name", body="@column_definitions"),
        Rule(
            ("create_view", "create_materialized_view"),
            Role.TYPE,
            _TYPE,
            name="@object_reference/name",
            body="@create_query",
        ),
        Rule(("create_type",), Role.TYPE, _TYPE, name="@object_reference/name", body=None),
        Rule(("column_definition",), Role.FIELD, body=None),
    ),
    calls=(CallRule(("invocation",), callee="@object_reference/name|#0", arguments="."),),
    block_kinds=frozenset(),
)

CMAKE = LanguageSpec(
    language="cmake",
    family="cmake",
    rules=(
        Rule(
            ("function_def",),
            Role.CALLABLE,
            name="@function_command/@argument_list/#0",
            params="@function_command/@argument_list",
            params_offset=1,
            body="@body",
        ),
        Rule(
            ("macro_def",),
            Role.CALLABLE,
            name="@macro_command/@argument_list/#0",
            params="@macro_command/@argument_list",
            params_offset=1,
            body="@body",
        ),
    ),
    calls=(CallRule(("normal_command",), callee="@identifier", arguments="@argument_list"),),
    block_kinds=frozenset({"body"}),
    call_exclusions=frozenset(
        {
            "set",
            "message",
            "if",
            "endif",
            "else",
            "foreach",
            "endforeach",
            "include",
            "project",
            "cmake_minimum_required",
        }
    ),
)

GRAPHQL = LanguageSpec(
    language="graphql",
    family="graphql",
    rules=(
        Rule(
            ("object_type_definition",),
            Role.TYPE,
            _CLASS,
            name="@name",
            supertypes=("@implements_interfaces",),
            body="@fields_definition",
        ),
        Rule(
            ("interface_type_definition",),
            Role.TYPE,
            _INTERFACE,
            name="@name",
            supertypes=("@implements_interfaces",),
            body="@fields_definition",
        ),
        Rule(("input_object_type_definition",), Role.TYPE, _STRUCT, name="@name", body="@input_fields_definition"),
        Rule(("enum_type_definition",), Role.TYPE, _ENUM, name="@name", body="@enum_values_definition"),
        Rule(("enum_value_definition",), Role.CONSTANT, name="@enum_value", body=None),
        Rule(("union_type_definition", "scalar_type_definition"), Role.TYPE, _TYPE, name="@name", body=None),
        Rule(("field_definition", "input_value_definition"), Role.FIELD, name="@name", body=None, scope="member"),
    ),
    type_ref_kinds=frozenset({"named_type"}),
    builtin_types=frozenset({"Int", "Float", "String", "Boolean", "ID"}),
    block_kinds=frozenset(),
)

HCL = LanguageSpec(
    language="hcl",
    family="hcl",
    rules=(
        Rule(("block",), Role.MODULE, names="@*", name_join=".", body="@body"),
        Rule(("attribute",), Role.FIELD, name="@identifier", body="@expression"),
    ),
    calls=(CallRule(("function_call",), callee="@identifier", arguments="@function_arguments"),),
    block_kinds=frozenset(),
)

TYPST = LanguageSpec(
    language="typst",
    family="typst",
    rules=(
        Rule(
            ("let",),
            Role.CALLABLE,
            name="pattern/item",
            params="pattern/@group",
            body="value",
            require=("pattern", "call"),
        ),
        Rule(("let",), Role.FIELD, name="pattern", body="value", require=("pattern", "ident")),
        Rule(("import",), Role.IMPORT, name="#0", body=None),
    ),
    calls=(CallRule(("call",), callee="item", arguments="@group"),),
    member_access={"field": ("#0", "field")},
    block_kinds=frozenset(),
    binding_paths=(("group", "@ident*"),),
)

VIM = LanguageSpec(
    language="vim",
    family="vim",
    rules=(
        Rule(
            ("function_definition",),
            Role.CALLABLE,
            name="@function_declaration/name",
            params="@function_declaration/parameters",
            body="@body",
        ),
        Rule(("let_statement",), Role.FIELD, name="#0", body=None, scope="top"),
        Rule(("command_statement",), Role.CALLABLE, name="name", params=None, body="repl"),
        Rule(("source_statement", "runtime_statement"), Role.IMPORT, name="#0", body=None),
    ),
    calls=(CallRule(("call_expression",), callee="function", arguments="argument+"),),
    member_access={"scoped_identifier": ("scope", "@identifier")},
    block_kinds=frozenset({"body"}),
    binding_paths=(("parameters", "@identifier*"), ("let_statement", "#0")),
)

WAT = LanguageSpec(
    language="wat",
    family="wat",
    rules=(
        Rule(("module",), Role.MODULE, name="identifier", default_name="module", body="."),
        Rule(("module_field_func",), Role.CALLABLE, name="identifier", params="@func_type_params", body="@instr_list"),
        Rule(
            ("module_field_global", "module_field_memory", "module_field_table"),
            Role.FIELD,
            name="identifier",
            body=None,
        ),
        Rule(("module_field_type",), Role.TYPE, _TYPE, name="identifier", body=None),
        Rule(("module_field_import",), Role.IMPORT, name="#0", body=None),
    ),
    calls=(
        CallRule(
            ("instr_plain",),
            callee="@index/@identifier",
            arguments=None,
            when=("@op_index", frozenset({"call", "return_call", "ref.func"})),
        ),
    ),
    block_kinds=frozenset(),
)

MAKE = LanguageSpec(
    language="make",
    family="make",
    rules=(
        Rule(("rule",), Role.CALLABLE, name="@targets/@word", params=None, body="@recipe"),
        Rule(("variable_assignment",), Role.FIELD, body=None, scope="top"),
        Rule(("include_directive",), Role.IMPORT, name="#0", body=None),
    ),
    calls=(CallRule(("prerequisites",), callee="@word", arguments=None),),
    block_kinds=frozenset(),
)

DOCKERFILE = LanguageSpec(
    language="dockerfile",
    family="dockerfile",
    rules=(Rule(("from_instruction",), Role.IMPORT, name="@image_spec/name", body=None),),
)

CSS = LanguageSpec(language="css", family="css")

SCSS = LanguageSpec(
    language="scss",
    family="css",
    rules=(
        Rule(("use_statement", "import_statement", "forward_statement"), Role.IMPORT, name="@string_value", body=None),
        Rule(
            ("mixin_statement", "function_statement"), Role.CALLABLE, name="@name", params="@parameters", body="@block"
        ),
        Rule(
            ("declaration",),
            Role.FIELD,
            name="@variable_name",
            body=None,
            scope="top",
            require=("@variable_name", "variable_name"),
        ),
    ),
    calls=(
        CallRule(("include_statement",), callee="@identifier", arguments="@arguments"),
        CallRule(("call_expression",), callee="@function_name", arguments="@arguments"),
    ),
    block_kinds=frozenset(),
    binding_paths=(("parameter", "@variable_name"),),
    call_exclusions=frozenset(
        {"var", "calc", "url", "rgb", "rgba", "hsl", "hsla", "map-get", "if", "nth", "unquote", "darken", "lighten"}
    ),
)

JSONNET = LanguageSpec(
    language="jsonnet",
    family="jsonnet",
    rules=(
        Rule(("bind",), Role.CALLABLE, name="function", params="params", body="body", require=("function", "id")),
        Rule(("bind",), Role.FIELD, name="@id|#0", body="#-1"),
        Rule(("field",), Role.FIELD, name="#0", body="#-1"),
        Rule(("import",), Role.IMPORT, name="@string", body=None),
    ),
    calls=(CallRule(("functioncall",), callee="#0", arguments="@args"),),
    member_access={"fieldaccess": ("#0", "#-1")},
    block_kinds=frozenset(),
    binding_paths=(("param", "identifier"),),
)

JUST = LanguageSpec(
    language="just",
    family="just",
    rules=(
        Rule(
            ("recipe",),
            Role.CALLABLE,
            name="@recipe_header/name",
            params="@recipe_header/@parameters",
            body="@recipe_body",
        ),
        Rule(("assignment",), Role.FIELD, name="left", body=None),
        Rule(("import", "module"), Role.IMPORT, name="#0", body=None),
    ),
    calls=(CallRule(("dependency",), callee="name", arguments=None),),
    block_kinds=frozenset(),
    binding_paths=(("parameter", "name"),),
)

BATCH = LanguageSpec(
    language="batch",
    family="batch",
    rules=(Rule(("label",), Role.CALLABLE, name=".", params=None, body=None),),
    calls=(CallRule(("call_stmt",), callee="@argument_list/#0|#0", arguments=None),),
    block_kinds=frozenset(),
)

GOTMPL = LanguageSpec(
    language="gotmpl",
    family="gotmpl",
    rules=(Rule(("define_action", "block_action"), Role.CALLABLE, params=None, body=None),),
    calls=(CallRule(("template_action",), callee="name", arguments=None),),
    block_kinds=frozenset(),
)

JINJA2 = LanguageSpec(
    language="jinja2",
    family="jinja2",
    calls=(CallRule(("fn_call",), callee="fn_name", arguments="argument_list"),),
    block_kinds=frozenset(),
)

HEEX = LanguageSpec(
    language="heex",
    family="elixir",
    calls=(CallRule(("component_name",), callee="@function", arguments=None),),
)

SVELTE = LanguageSpec(language="svelte", family="js")
VUE = LanguageSpec(language="vue", family="js")
ASTRO = LanguageSpec(language="astro", family="js")

#: Every grammar with a spec, keyed by the language name `detect_language` reports.
SPECS: dict[str, LanguageSpec] = {
    spec.language: spec
    for spec in (
        JAVASCRIPT,
        TYPESCRIPT,
        TSX,
        PYTHON,
        STARLARK,
        GO,
        RUST,
        C,
        CPP,
        CSHARP,
        RUBY,
        PHP,
        KOTLIN,
        SWIFT,
        SCALA,
        DART,
        LUA,
        ZIG,
        ELIXIR,
        ERLANG,
        HASKELL,
        OCAML,
        JULIA,
        R,
        PERL,
        BASH,
        POWERSHELL,
        SOLIDITY,
        GROOVY,
        OBJC,
        CLOJURE,
        SCHEME,
        RACKET,
        NIX,
        FORTRAN,
        SQL,
        CMAKE,
        GRAPHQL,
        HCL,
        TYPST,
        VIM,
        WAT,
        MAKE,
        DOCKERFILE,
        CSS,
        SCSS,
        JSONNET,
        JUST,
        BATCH,
        GOTMPL,
        JINJA2,
        HEEX,
        SVELTE,
        VUE,
        ASTRO,
    )
}


def spec_for(language: str | None) -> LanguageSpec | None:
    """The spec of a language, or None when the language has none."""
    if language is None:
        return None
    return SPECS.get(language)


def is_test_file(relative_path: str) -> bool:
    """Whether a file's NAME marks it as a test source in its language (`test_x.py`, `x_test.go`)."""
    name = relative_path.rsplit("/", 1)[-1]
    spec = spec_for(detect_language(PurePosixPath(name)))
    return spec is not None and any(fnmatch(name, pattern) for pattern in spec.test_file_patterns)


def family_of(language: str | None) -> str | None:
    """The resolution family of a language: the languages whose symbols it may resolve into."""
    spec = spec_for(language)
    return spec.family if spec is not None else language


__all__ = ["SPECS", "family_of", "is_test_file", "spec_for"]
