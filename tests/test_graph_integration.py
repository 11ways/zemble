"""One slow end-to-end journey over a real Java workspace.

Skipped unless the javaweb checkout is present. Every fact asserted here was read
out of the source first; they are not guesses about what the graph might find.
"""

from pathlib import Path

import pytest

from zemble.graph.provider import SqliteGraphProvider
from zemble.graph.store import build_graph

ZENIT = Path("/home/skerit/projects/javaweb/zenit")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not ZENIT.is_dir(), reason="the javaweb workspace is not checked out here"),
]


@pytest.fixture(scope="module")
def zenit_graph(tmp_path_factory: pytest.TempPathFactory) -> SqliteGraphProvider:
    """Build the graph for the zenit repository once for this module."""
    import os

    cache = tmp_path_factory.mktemp("zenit-graph-cache")
    previous = os.environ.get("ZEMBLE_CACHE_LOCATION")
    os.environ["ZEMBLE_CACHE_LOCATION"] = str(cache)
    try:
        build_graph(str(ZENIT))
        provider = SqliteGraphProvider(str(ZENIT))
        yield provider
        provider.close()
    finally:
        if previous is None:
            os.environ.pop("ZEMBLE_CACHE_LOCATION", None)
        else:
            os.environ["ZEMBLE_CACHE_LOCATION"] = previous


def _files(hits) -> set[str]:
    """Collect the file names a hit list points at."""
    return {Path(hit.symbol.file_path).name for hit in hits}


def test_real_workspace_journey(zenit_graph: SqliteGraphProvider) -> None:
    """Six facts read out of the zenit source, answered by the graph."""
    # 1. Pagination arithmetic: PageWindow.of is called by the record source handler
    #    and by the static data provider.
    page_window = zenit_graph.definition("PageWindow.of")
    assert len(page_window) == 1, "step 1: PageWindow.of is declared once"
    callers = _files(zenit_graph.callers(page_window[0].id))
    assert {"RecordSourceHandlers.java", "StaticDataProvider.java"} <= callers, "step 1: both callers are found"

    # 2. Conditional GET: EntityTags.matchesIfNoneMatch is called by ConditionalRequest
    #    and by BrandEndpoints.
    entity_tags = zenit_graph.definition("EntityTags.matchesIfNoneMatch")
    assert len(entity_tags) == 1, "step 2: the EntityTags overload is pinned down"
    assert {"ConditionalRequest.java", "BrandEndpoints.java"} <= _files(zenit_graph.callers(entity_tags[0].id)), (
        "step 2: both callers are found"
    )

    # 3. Object storage: LocalDiskStorageAdapter is the only shipped StorageAdapter in zenit.
    adapter = [symbol for symbol in zenit_graph.definition("StorageAdapter") if symbol.kind.value == "interface"]
    implementations = _files(zenit_graph.implementations(adapter[0].id))
    assert implementations == {"LocalDiskStorageAdapter.java"}, "step 3: exactly one adapter ships in zenit"

    # 4. UI preference cookies: the three shipped preferences all go through PreferenceCookie.named.
    named = zenit_graph.definition("PreferenceCookie.named")
    users = _files(zenit_graph.callers(named[0].id))
    assert {"Themes.java", "Timezones.java", "Disclosures.java"} <= users, "step 4: all three preferences are found"

    # 5. Text helpers: Slugs is used by the sluggable behaviour and by the dev tunnel boot.
    slugs = [symbol for symbol in zenit_graph.definition("Slugs") if symbol.kind.value == "class"]
    referencing = _files(zenit_graph.references(slugs[0].id))
    assert {"SluggableBehaviour.java", "DevTunnelBoot.java"} <= referencing, "step 5: both users are found"

    # 6. Test edges: SluggableBehaviourTest is found as the test of SluggableBehaviour.
    behaviour = [
        symbol for symbol in zenit_graph.definition("SluggableBehaviour") if symbol.kind.value in ("class", "interface")
    ]
    assert "SluggableBehaviourTest.java" in _files(zenit_graph.tests_of(behaviour[0].id)), (
        "step 6: the naming-based test edge is found"
    )
