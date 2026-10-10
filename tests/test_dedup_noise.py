"""One fixture per false-positive rule of the holed, idiom, re-implementation and vocabulary channels.

Each test writes a tiny tree, checks the rule takes the noise out, and checks its near miss still reports.
"""

from pathlib import Path

from tests.test_dedup import BagOfWordsEmbedder
from zemble.dedup.detect import DupeOptions, find_duplication
from zemble.dedup.model import CloneKind
from zemble.dedup.settings import SettingKey, ShapeSettings
from zemble.home.config import HomeConfig


def _write(root: Path, files: dict[str, str]) -> Path:
    """Write a tree of Java sources under `root`."""
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def _run(root: Path, kind: CloneKind):
    """Scan one tree for one channel."""
    return find_duplication(root, DupeOptions(kinds=(kind,), jobs=1), embedder=BagOfWordsEmbedder())


def _texts(report) -> list[str]:
    """Every class's reasons, one line per class."""
    return [" | ".join(clone.wire_reasons) for clone in report.classes]


def _names(report) -> set[str]:
    """Every member name of every class."""
    return {member.name for clone in report.classes for member in clone.members}


def test_holed_values_only_groups_are_no_copies(tmp_path: Path) -> None:
    """Delegating constructors, value declarations and the language applied to other data are not holed copies."""
    files = {}
    for index, (prefix, cron) in enumerate((("a-", "0 1 * * *"), ("b-", "0 2 * * *"), ("c-", "0 3 * * *"))):
        files[f"T{index}.java"] = (
            f"class T{index} {{\n"
            f"  T{index}() {{ this(new Service()); }}\n"
            f'  List<Object> schedules() {{ return Roles.when(List.of(Decl.fallback("{cron}")), R{index}); }}\n'
            f'  String alias(String v) {{ return "{prefix}" + v; }}\n'
            f"  String tail(String path) {{ return path.substring(path.lastIndexOf('{'/.:'[index]}') + 1); }}\n"
            f"  String name(Row row) {{ return Texts.label(row.get(Model.NAME)).trim(); }}\n"
            "}\n"
        )
    report = _run(_write(tmp_path, files), CloneKind.HOLED)
    names = _names(report)

    # 1. `this(new Service())` binds a default; each `schedules()` declares its own value; `"x" + v` and
    #    `substring(lastIndexOf(c))` are the language applied to other data.
    for member in ("T0.T0", "T0.schedules", "T0.alias", "T0.tail"):
        assert member not in names, f"step 1: {member} is no copy"

    # 2. Near miss: the same helper holding the same values in three files is a copy.
    assert {"T0.name", "T1.name", "T2.name"} <= names, "step 2: one helper written three times"


def test_idiom_language_calls_are_no_idiom_links_and_roots_name_the_shape(tmp_path: Path) -> None:
    """`String.valueOf((Object) x)` and `Boolean.TRUE.equals(x)` are the language; a domain wrapper is an idiom."""
    files = {}
    for index in range(4):
        files[f"P{index}.java"] = (
            f"class P{index} {{\n"
            f"  String a(Row r) {{ return String.valueOf((Object) r.get(Model.NAME)); }}\n"
            f"  boolean b(Row r) {{ return Boolean.TRUE.equals(r.get(Model.ACTIVE)); }}\n"
            f"  String c(Row r) {{ return Texts.valueOf((Object) r.get(Model.NAME)); }}\n"
            "}\n"
        )
    report = _run(_write(tmp_path, files), CloneKind.IDIOM)
    shapes = [clone.notes[0] for clone in report.classes]

    # 1. A JDK static call never counts toward the two calls an idiom chains.
    assert not any("String.valueOf" in shape or "Boolean.TRUE" in shape for shape in shapes), "step 1: language"

    # 2. Near miss: the same shape on a domain type is an idiom, rooted at its shape rather than a member.
    (clone,) = [clone for clone in report.classes if "Texts.valueOf" in clone.notes[0]]
    assert clone.root_label.startswith("wrapper Texts.valueOf((Object) $0.get("), "step 2: the root names the shape"
    assert clone.to_dict()["root"] == clone.root_label, "step 2: the wire carries it"


def test_frozen_code_takes_part_in_no_shape_kind(tmp_path: Path) -> None:
    """A migration repeats its DSL and writes values out by design; the same code elsewhere is reported."""
    body = '  void up(Schema s) {{ s.alterTable("t{n}", t -> t.addColumn("c{n}", Type.STRING).nullable(true)); }}\n'
    files = {f"app/migration/M{index}.java": f"class M{index} {{\n{body.format(n=index)}}}\n" for index in range(4)}
    files.update({f"app/live/L{index}.java": f"class L{index} {{\n{body.format(n=index)}}}\n" for index in range(4)})
    report = _run(_write(tmp_path, files), CloneKind.IDIOM)
    members = {member.file_path for clone in report.classes for member in clone.members}

    # 1. Nothing under `migration/` is a site.
    assert not any("/migration/" in path for path in members), "step 1: frozen code is out"

    # 2. Near miss: the live copies still form the idiom.
    assert any("/live/" in path for path in members), "step 2: live code still counts"
    assert ShapeSettings().is_frozen("db/migrations/V1.java") and not ShapeSettings().is_frozen("app/Migrations.java")


def test_vocabulary_copy_keys_derived_enums_and_prefixed_names(tmp_path: Path) -> None:
    """Catalog keys are no values, an enum built from constants re-expresses them, a name inside a value is none."""
    files = {
        "src/common/java/m/Device.java": (
            "class Device {\n"
            '  public static final String TYPE_DISK = "disk";\n'
            '  public static final String TYPE_NIC = "nic";\n'
            '  public static final String TYPE_CDROM = "cdrom";\n'
            '  public static final String KIND = "device-kind";\n'
            "}\n"
        ),
        "src/common/java/m/DeviceType.java": (
            "enum DeviceType { DISK(Device.TYPE_DISK), NIC(Device.TYPE_NIC), CDROM(Device.TYPE_CDROM); }\n"
        ),
        "src/common/java/m/Kinds.java": 'enum Kinds { DISK("disk"), NIC("nic"), CDROM("cdrom"); }\n',
        "src/common/java/m/Slugs.java": (
            "class Slugs {\n"
            '  public static final String INSTANCES = "instances";\n'
            '  public static final String SITES = "sites";\n'
            '  public static final String BANS = "bans";\n'
            "}\n"
        ),
        "src/common/java/m/Budget.java": (
            'enum Budget { INSTANCES("app:instances:"), SITES("app:sites:"), BANS("app:bans:"); }\n'
        ),
        "src/common/java/m/Labels.java": (
            "class Labels {\n"
            '  Object a() { return AppMicrocopy.FIELD.of("device-kind"); }\n'
            '  Object b() { return Forms.of("device-kind"); }\n'
            "}\n"
        ),
    }
    report = _run(_write(tmp_path, files), CloneKind.VOCABULARY)
    texts = _texts(report)

    # 1. `DeviceType` reads Device's constants: derived, so it never joins a family; `Kinds` restates them.
    family = [clone for clone in report.classes if "value sets declare" in clone.notes[0]]
    assert family and {member.name for member in family[0].members} == {"Device.TYPE_*", "Kinds"}, "step 1"

    # 2. `Budget` names its members after one token of its values: no "instances" set beside `Slugs`.
    assert not any("Budget" in {member.name for member in clone.members} for clone in report.classes), "step 2"

    # 3. A Microcopy key is no use of "device-kind"; another call's argument is.
    (kind,) = [clone for clone in report.classes if 'value "device-kind"' in clone.notes[0]]
    assert {member.name for member in kind.members} == {"Device.KIND", "Labels.b"}, "step 3: the key is left out"
    assert not any("Labels.a" in text for text in texts)


def test_vocabulary_reads_source_sets_and_sorts_by_score(tmp_path: Path) -> None:
    """A common use cannot read a server constant; every class is ordered by score."""
    files = {
        "src/server/java/s/Columns.java": 'class Columns { public static final String CPU = "cpu_limit"; }\n',
        "src/common/java/c/Shared.java": 'class Shared { Object a() { return Map.of("cpu_limit", 1); } }\n',
        "src/server/java/s/Api.java": 'class Api { Object a() { return Map.of("cpu_limit", 1); } }\n',
        "src/server/java/s/Other.java": 'class Other { Object a() { return Map.of("cpu_limit", 2); } }\n',
    }
    for index in range(3):
        files[f"src/common/java/c/S{index}.java"] = (
            f'class S{index} {{ public static final String STATUS_FAILED = "failed";'
            f' public static final String STATUS_DONE = "done"; public static final String STATUS_NEW = "new"; }}\n'
        )
    report = _run(_write(tmp_path, files), CloneKind.VOCABULARY)

    # 1. The server uses are listed beside the server constant; the common one cannot read it and is not.
    (cpu,) = [clone for clone in report.classes if 'value "cpu_limit"' in clone.notes[0]]
    assert {member.name for member in cpu.members} == {"Columns.CPU", "Api.a", "Other.a"}, "step 1"
    assert ShapeSettings().reaches("src/server/java/a.java", "src/common/java/b.java"), "step 1: server reads common"
    assert ShapeSettings().reaches("app/A.java", "src/server/java/b.java"), "step 1: an unknown fold says nothing"

    # 2. Classes come in one order, by score, whatever their flavour.
    scores = [clone.score for clone in report.classes]
    assert scores == sorted(scores, reverse=True) and len(scores) >= 2, "step 2: sorted by score"


def test_copy_keys_and_frozen_globs_are_configurable(tmp_path: Path) -> None:
    """`[dupes]` in home.toml replaces a default; an unknown key is noted, never applied."""
    (tmp_path / ".zemble").mkdir()
    (tmp_path / ".zemble" / "home.toml").write_text('[dupes]\ncopy_keys = ["Forms.of:0"]\nfrozen = ["*/legacy/*"]\n')
    settings, notes = ShapeSettings.load(tmp_path)
    assert not notes and settings.copy_keys == ("Forms.of:0",) and settings.is_frozen("app/legacy/A.java"), "step 1"
    assert settings.frozen != SettingKey.FROZEN.default, "step 1: the default is replaced"

    (tmp_path / ".zemble" / "home.toml").write_text('[dupes]\nfrozzen = ["x"]\n')
    settings, notes = ShapeSettings.load(tmp_path)
    assert notes and "frozzen" in notes[0] and settings == ShapeSettings(), "step 2: unknown key, defaults hold"
    assert HomeConfig.load(tmp_path).dupes == {"frozzen": ("x",)}, "step 2: home.toml itself parses the section"


def test_reimplements_standard_homes_rebinding_roles_and_receivers(tmp_path: Path) -> None:
    """JDK homes come first, a copy bound to another constant is demoted, role and instance APIs are no homes."""
    files = {
        "core/Texts.java": (
            "public class Texts {\n"
            '  public static String orEmpty(Object value) { return value == null ? "" : String.valueOf(value); }\n'
            "  public static boolean canConfig(Ctx ctx, Row row) {\n"
            "    Integer id = row.get(Model.OWNER_ID);\n"
            "    return id != null && Access.hasCapability(ctx, id, Caps.CONFIG) && Audit.note(ctx, id);\n"
            "  }\n"
            "  public static boolean canRead(Ctx ctx, Row row) {\n"
            "    Integer id = row.get(Model.OWNER_ID);\n"
            "    return id != null && Access.hasCapability(ctx, id, Caps.READ) && Audit.note(ctx, id);\n"
            "  }\n"
            '  @Registered(name = "trimmed")\n'
            "  public static String trimmedText(Object raw) { String text = Strings.cast(raw); return text.trim(); }\n"
            "}\n"
        ),
        "app/Pages.java": (
            "class Pages {\n"
            '  private static String text(Object value) { return value == null ? "" : String.valueOf(value); }\n'
            "  private static String trimOf(Object raw) { String text = Strings.cast(raw); return text.trim(); }\n"
            "  private static boolean canManage(Ctx ctx, Row row) {\n"
            "    Integer id = row.get(Model.OWNER_ID);\n"
            "    return id != null && Access.hasCapability(ctx, id, Caps.MANAGE) && Audit.note(ctx, id);\n"
            "  }\n"
            "  private static boolean mayRead(Ctx ctx, Row row) {\n"
            "    Integer id = row.get(Model.OWNER_ID);\n"
            "    return id != null && Access.hasCapability(ctx, id, Caps.READ) && Audit.note(ctx, id);\n"
            "  }\n"
            "}\n"
        ),
    }
    report = _run(_write(tmp_path, files), CloneKind.REIMPLEMENTS)
    by_first = {clone.members[0].name: clone for clone in report.classes}

    # 1. `Pages.text` and `Texts.orEmpty` are `Objects.toString(value, "")`: the JDK is the home, not `Texts`.
    standard = by_first["Pages.text"]
    assert "Objects.toString(value, fallback); call it" in standard.notes[0], "step 1: the JDK call"
    assert {member.name for member in standard.members} == {"Pages.text", "Texts.orEmpty"}, "step 1: both bodies"

    # 2. `canManage` binds MANAGE where `canConfig` binds CONFIG: demoted below every other class.
    demoted = [clone for clone in report.classes if clone.demoted]
    assert demoted and "extract the constant as a parameter" in demoted[0].notes[0], "step 2: demoted, re-worded"
    assert report.classes[-1].demoted, "step 2: ranked last"

    # 3. Near miss: `mayRead` holds READ like `canRead`: a plain copy, told to call it.
    assert any(
        clone.members[0].name == "Pages.mayRead" and not clone.demoted and "call Texts.canRead" in clone.notes[0]
        for clone in report.classes
    ), "step 3: same constants, a copy"

    # 4. An API a framework registers by annotation is a role: `trimOf` is not sent to `trimmedText`.
    assert "Pages.trimOf" not in by_first, "step 4: an annotated member is no home"


def test_reimplements_needs_an_instance_for_an_instance_api(tmp_path: Path) -> None:
    """A static helper cannot call a method that needs an instance it does not have, whatever it converts."""
    from zemble.dedup.reimplements import _substitution
    from zemble.dedup.units import extract_file

    source = (
        "public class Task {\n"
        "  private Object settings;\n"
        "  public Object getSettings() { return settings == null ? java.util.Map.of() : settings; }\n"
        "  public static Object settingsOf(Object raw) { return raw == null ? java.util.Map.of() : raw; }\n"
        "  public static Object fromTask(Task task) {\n"
        "    return task == null ? java.util.Map.of() : task.getSettings();\n"
        "  }\n"
        "}\n"
    )
    shaped = {unit.name: unit for unit in extract_file(source.encode(), "Task.java", shaped=True).shaped}
    assert _substitution(shaped["Task.settingsOf"], shaped["Task.getSettings"]) == 0.0, "step 1: no instance"
    assert _substitution(shaped["Task.fromTask"], shaped["Task.getSettings"]) == 1.0, "step 2: a Task parameter is one"
