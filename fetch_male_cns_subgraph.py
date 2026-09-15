"""Fetch the small real-connectome slice used by the runner experiment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
TYPES = ("R1-R6", "L1", "L2", "L3", "DNc01", "DNc02")
TYPE_LIST = ", ".join(repr(neuron_type) for neuron_type in TYPES)

NODE_QUERY = f"""
MATCH (n:Neuron)
WHERE n.type IN [{TYPE_LIST}]
RETURN n.type AS type,
       n.superclass AS superclass,
       n.class AS neuron_class,
       count(n) AS neuron_count,
       sum(n.pre) AS total_pre,
       sum(n.post) AS total_post
ORDER BY type
""".strip()

EDGE_QUERY = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
WHERE a.type IN [{TYPE_LIST}] AND b.type IN [{TYPE_LIST}]
RETURN a.type AS upstream_type,
       b.type AS downstream_type,
       sum(c.weight) AS total_weight,
       count(*) AS body_edges
ORDER BY upstream_type, total_weight DESC, downstream_type
""".strip()


def query(cypher: str) -> list[dict[str, object]]:
    payload = json.dumps({"cypher": cypher, "dataset": DATASET}).encode()
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": "fly-brain-runner/0.1",
    }
    token = os.environ.get("NEUPRINT_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(
        f"{SERVER}/api/custom/custom", data=payload, headers=headers, method="POST"
    )
    try:
        with urlopen(request, timeout=60) as response:
            result = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"neuPrint request failed: {exc}") from exc

    columns = result.get("columns")
    rows = result.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise RuntimeError(f"unexpected neuPrint response: {result!r}")
    return [dict(zip(columns, row)) for row in rows]


def build_manifest() -> dict[str, object]:
    nodes = query(NODE_QUERY)
    edges = query(EDGE_QUERY)
    found_types = {row.get("type") for row in nodes}
    if found_types != set(TYPES):
        raise RuntimeError(f"expected metadata for {TYPES}, got {sorted(found_types)!r}")
    if not edges:
        raise RuntimeError("the selected types produced no connectome edges")

    return {
        "schema_version": 1,
        "source": {
            "dataset": DATASET,
            "server": SERVER,
            "api": f"{SERVER}/api/custom/custom",
            "download_page": "https://male-cns.janelia.org/download/",
            "retrieved_on": date.today().isoformat(),
        },
        "selection": {
            "types": list(TYPES),
            "reason": (
                "A deliberately small visual-to-descending slice: R1-R6 is the "
                "visual input class, L1/L2/L3 are optic-lobe intrinsic types, "
                "and DNc01/DNc02 are descending output types."
            ),
            "aggregation": "one node per published neuron type; edges are summed by type pair",
        },
        "queries": {"nodes": NODE_QUERY, "edges": EDGE_QUERY},
        "nodes": nodes,
        "edges": edges,
        "limitations": [
            "This is a type-level reduction, not the complete neuron-by-neuron graph.",
            "The manifest contains wiring statistics, not a full neural simulator.",
            "Mapping pixels to R1-R6 and mapping descending activity to game actions are engineering choices.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data/male_cns_visual_subgraph.json",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {len(manifest['nodes'])} types and {len(manifest['edges'])} edges to {args.output}")


if __name__ == "__main__":
    main()
