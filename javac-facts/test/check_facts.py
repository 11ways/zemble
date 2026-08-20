#!/usr/bin/env python3
"""Assert that the facts emitted for the test fixtures match the graph-facts contract."""

import json
import sys

# Facts that must be present, as subsets of an emitted object.
EXPECTED = [
    # Overloads are resolved by javac, not by name.
    {"t": "call", "from": "demo.Demo#overloadCalls()", "to": "demo.Demo#bar(int)"},
    {"t": "call", "from": "demo.Demo#overloadCalls()", "to": "demo.Demo#bar(java.lang.String)"},
    {"t": "call", "from": "demo.Demo#overloadCalls()", "to": "demo.Demo#bar(java.lang.Object)"},
    # Varargs erase to an array parameter.
    {"t": "symbol", "ref": "demo.Demo#sum(int[])", "kind": "method"},
    {"t": "call", "from": "demo.Demo#overloadCalls()", "to": "demo.Demo#sum(int[])"},
    # A method reference is a call.
    {"t": "call", "from": "demo.Demo#lambdas()", "to": "demo.Demo#describe()"},
    # A lambda body's calls belong to the enclosing method.
    {"t": "call", "from": "demo.Demo#lambdas()", "to": "demo.Demo#bar(int)"},
    # Generics: a bounded type variable erases to its bound, both in the signature and the call.
    {"t": "symbol", "ref": "demo.Demo#total(java.util.List)", "kind": "method"},
    {"t": "call", "from": "demo.Demo#total(java.util.List)", "to": "java.lang.Number#doubleValue()"},
    # Calls into the JDK, including through a static import.
    {"t": "call", "from": "demo.Demo#library()", "to": "java.util.List#add(java.lang.Object)"},
    {"t": "call", "from": "demo.Demo#library()", "to": "java.util.Arrays#asList(java.lang.Object[])"},
    {"t": "call", "from": "demo.Demo#library()", "to": "java.util.ArrayList#<init>(java.util.Collection)"},
    # this(...) and super(...).
    {"t": "call", "from": "demo.Demo#<init>()", "to": "demo.Demo#<init>(int)"},
    {"t": "call", "from": "demo.Demo#<init>(int)", "to": "demo.Base#<init>(int)"},
    {"t": "call", "from": "demo.Demo#describe()", "to": "demo.Base#describe()"},
    # Anonymous classes: flat names, hierarchy edges and members.
    {"t": "symbol", "ref": "demo.Demo$1", "kind": "class"},
    {"t": "extends", "from": "demo.Demo$1", "to": "demo.Base"},
    {"t": "symbol", "ref": "demo.Demo$2", "kind": "class"},
    {"t": "implements", "from": "demo.Demo$2", "to": "demo.Greeter"},
    {"t": "override", "from": "demo.Demo$2#greet(java.lang.String)", "to": "demo.Greeter#greet(java.lang.String)"},
    {"t": "call", "from": "demo.Demo#inner()", "to": "demo.Demo$1#<init>(int)"},
    # Local classes carry javac's flat name too.
    {"t": "symbol", "ref": "demo.Demo$1Local#render()", "kind": "method"},
    {"t": "call", "from": "demo.Demo#inner()", "to": "demo.Demo$1Local#render()"},
    # Nested classes stay source-style dotted.
    {"t": "symbol", "ref": "demo.Demo.Nested.Deeper#ping()", "kind": "method"},
    {"t": "call", "from": "demo.Demo.Nested#call()", "to": "demo.Demo.Nested.Deeper#ping()"},
    # Field initialisers and initialiser blocks.
    {"t": "call", "from": "demo.Demo#<instance-init>()", "to": "java.util.ArrayList#<init>()"},
    {"t": "call", "from": "demo.Demo#<clinit>()", "to": "java.io.PrintStream#println(java.lang.String)"},
    # Interface default method calling an abstract sibling.
    {"t": "call", "from": "demo.Greeter#greetLoudly(java.lang.String)", "to": "demo.Greeter#greet(java.lang.String)"},
    # Records.
    {"t": "symbol", "ref": "demo.Point", "kind": "record"},
    {"t": "implements", "from": "demo.Point", "to": "demo.Named"},
    {"t": "symbol", "ref": "demo.Point#<init>(int,int)", "kind": "constructor"},
    {"t": "override", "from": "demo.Point#name()", "to": "demo.Named#name()"},
    # Enums with constant bodies.
    {"t": "symbol", "ref": "demo.Level", "kind": "enum"},
    {"t": "symbol", "ref": "demo.Level#LOW", "kind": "field"},
    {"t": "extends", "from": "demo.Level$1", "to": "demo.Level"},
    {"t": "override", "from": "demo.Level$1#weight()", "to": "demo.Level#weight()"},
    # Annotation with constant arguments only.
    {
        "t": "annotation",
        "ref": "demo.Demo",
        "name": "demo.Marker",
        "args": {"value": "demo", "count": 3, "strict": True, "level": "HIGH", "tags": ["a", "b"]},
    },
]

# Facts that must NOT be present.
FORBIDDEN = [
    # An overload must never be reported by name alone.
    {"t": "call", "from": "demo.Demo#overloadCalls()", "to": "demo.Demo#bar()"},
    # java.lang.Object is not an interesting supertype.
    {"t": "extends", "from": "demo.Base", "to": "java.lang.Object"},
]


def matches(fact, expected):
    return all(fact.get(key) == value for key, value in expected.items())


def main(path):
    problems = []
    facts = []

    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.read().splitlines()

    if not lines:
        print("no output produced")
        return 1

    for number, line in enumerate(lines, start=1):
        try:
            facts.append(json.loads(line))
        except json.JSONDecodeError as error:
            problems.append("line %d is not valid JSON: %s" % (number, error))
            facts.append(None)

    header = facts[0]
    if not isinstance(header, dict) or header.get("zemble_facts") != 1:
        problems.append("line 1 is not a zemble_facts header")
    else:
        for key in ("tool", "tool_version", "generated_at", "language", "root"):
            if not header.get(key):
                problems.append("header is missing %s" % key)
        if header.get("tool") != "zemble-javac-facts":
            problems.append("header tool is %r" % header.get("tool"))
        if header.get("language") != "java":
            problems.append("header language is %r" % header.get("language"))

    for fact in facts[1:]:
        if isinstance(fact, dict) and fact.get("zemble_facts"):
            problems.append("a second header line was emitted")
            break

    # Every file line comes before the facts about that file, and appears once.
    declared = set()
    for number, fact in enumerate(facts[1:], start=2):
        if not isinstance(fact, dict):
            continue
        if fact.get("t") == "file":
            if fact["path"] in declared:
                problems.append("duplicate file line for %s (line %d)" % (fact["path"], number))
            if not fact.get("sha256") or len(fact["sha256"]) != 64:
                problems.append("bad sha256 on line %d" % number)
            declared.add(fact["path"])
        elif "path" in fact and fact["path"] not in declared:
            problems.append("line %d references undeclared file %s" % (number, fact["path"]))

    real = [fact for fact in facts if isinstance(fact, dict)]

    for expected in EXPECTED:
        if not any(matches(fact, expected) for fact in real):
            problems.append("missing fact: %s" % json.dumps(expected, sort_keys=True))

    for forbidden in FORBIDDEN:
        if any(matches(fact, forbidden) for fact in real):
            problems.append("forbidden fact present: %s" % json.dumps(forbidden, sort_keys=True))

    # Every symbol/call fact must carry a positive line number.
    for number, fact in enumerate(facts, start=1):
        if isinstance(fact, dict) and fact.get("t") in ("symbol", "call"):
            if not isinstance(fact.get("line"), int) or fact["line"] < 1:
                problems.append("line %d has no usable line number" % number)

    if problems:
        for problem in problems:
            print("FAIL: %s" % problem)
        print("%d problem(s) in %d fact line(s)" % (len(problems), len(facts) - 1))
        return 1

    print("OK: %d fact lines, %d files, %d assertions" % (len(facts) - 1, len(declared), len(EXPECTED) + len(FORBIDDEN)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "build/test/facts.jsonl"))
