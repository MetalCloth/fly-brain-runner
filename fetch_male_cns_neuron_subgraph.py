"""Fetch a small neuron-level MaleCNS subgraph for the runner experiment."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SERVER = "https://neuprint.janelia.org"
DATASET = "male-cns:v1.0"
TYPES = ("R1-R6", "L1", "L2", "L3", "DNc01", "DNc02")
TYPE_LIST = ", ".join(repr(neuron_type) for neuron_type in TYPES)

EDGE_QUERY = f"""
MATCH (a:Neuron)-[c:ConnectsTo]->(b:Neuron)
WHERE a.type IN [{TYPE_LIST}] AND b.type IN [{TYPE_LIST}]
RETURN a.bodyId AS upstream_body_id,
       a.type AS upstream_type,
       b.bodyId AS downstream_body_id,
       b.type AS downstream_type,
       c.weight AS weight
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
        with urlopen(request, timeout=120) as response:
            result = json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"neuPrint request failed: {exc}") from exc

    columns = result.get("columns")
    rows = result.get("data")
    if not isinstance(columns, list) or not isinstance(rows, list):
        raise RuntimeError(f"unexpected neuPrint response: {result!r}")
    return [dict(zip(columns, row)) for row in rows]


def select_nodes(
    edges: list[dict[str, object]], max_neurons_per_type: int
) -> tuple[set[int], dict[int, str], dict[int, float]]:
    body_types: dict[int, str] = {}
    scores: defaultdict[int, float] = defaultdict(float)
    parents: defaultdict[int, set[int]] = defaultdict(set)
    for edge in edges:
        source = int(edge["upstream_body_id"])
        target = int(edge["downstream_body_id"])
        weight = float(edge["weight"])
        source_type = str(edge["upstream_type"])
        target_type = str(edge["downstream_type"])
        body_types[source] = source_type
        body_types[target] = target_type
        scores[source] += weight
        scores[target] += weight
        parents[target].add(source)

    # Keep a real sensory-to-descending route before filling the remainder by
    # weight. Two reverse hops cover R1-R6 -> L1/L2/L3 -> DNc01/DNc02.
    required: set[int] = {
        body for body, neuron_type in body_types.items() if neuron_type.startswith("DNc")
    }
    frontier = set(required)
    for _ in range(2):
        frontier = {parent for target in frontier for parent in parents[target]}
        required.update(frontier)

    selected: set[int] = set()
    for neuron_type in TYPES:
        candidates = [body for body, kind in body_types.items() if kind == neuron_type]
        candidates.sort(key=lambda body: (-scores[body], body))
        forced = [body for body in candidates if body in required]
        if neuron_type.startswith("DNc"):
            keep = candidates
        else:
            remaining = [body for body in candidates if body not in required]
            keep = forced + remaining[: max(0, max_neurons_per_type - len(forced))]
        selected.update(keep)
    return selected, body_types, scores


def build_manifest(max_neurons_per_type: int) -> dict[str, object]:
    if max_neurons_per_type < 1:
        raise ValueError("max_neurons_per_type must be positive")

    all_edges = query(EDGE_QUERY)
    selected, body_types, scores = select_nodes(all_edges, max_neurons_per_type)
    if {body_types[body] for body in selected} != set(TYPES):
        raise RuntimeError("selection did not retain all six neuron types")

    selected_edges = [
        {
            "upstream_body_id": int(edge["upstream_body_id"]),
            "upstream_type": str(edge["upstream_type"]),
            "downstream_body_id": int(edge["downstream_body_id"]),
            "downstream_type": str(edge["downstream_type"]),
            "weight": float(edge["weight"]),
        }
        for edge in all_edges
        if int(edge["upstream_body_id"]) in selected
        and int(edge["downstream_body_id"]) in selected
    ]
    if not selected_edges:
        raise RuntimeError("the selected neurons produced no internal edges")

    body_list = ", ".join(str(body) for body in sorted(selected))
    node_query = f"""
MATCH (n:Neuron)
WHERE n.bodyId IN [{body_list}]
RETURN n.bodyId AS body_id,
       n.type AS type,
       n.instance AS instance,
       n.somaSide AS soma_side,
       n.pre AS pre,
       n.post AS post
ORDER BY type, body_id
""".strip()
    nodes = query(node_query)
    found = {int(node["body_id"]) for node in nodes}
    if found != selected:
        raise RuntimeError(f"node metadata mismatch: expected {len(selected)}, got {len(found)}")

    counts = {neuron_type: 0 for neuron_type in TYPES}
    for node in nodes:
        counts[str(node["type"])] += 1
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
            "max_neurons_per_type": max_neurons_per_type,
            "method": (
                "retain the two-hop sensory-to-DNc route first, then rank by the "
                "sum of internal selected-slice edge weights; retain all DNc neurons"
            ),
            "nodes_before_reduction": len(body_types),
            "edges_before_reduction": len(all_edges),
            "nodes_by_type": counts,
        },
        "queries": {"edges": EDGE_QUERY, "nodes": node_query},
        "nodes": sorted(nodes, key=lambda node: (str(node["type"]), int(node["body_id"]))),
        "edges": sorted(
            selected_edges,
            key=lambda edge: (
                edge["upstream_type"],
                edge["upstream_body_id"],
                edge["downstream_type"],
                edge["downstream_body_id"],
            ),
        ),
        "limitations": [
            "This is a selected neuron-level subgraph, not the complete connectome.",
            "The selection favors neurons with strong internal edge weight in this slice.",
            "Pixel-to-neuron and descending-neuron-to-action mappings remain engineering choices.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-neurons-per-type", type=int, default=48)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "data/male_cns_neuron_subgraph.json",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.max_neurons_per_type)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"saved {len(manifest['nodes'])} neurons and {len(manifest['edges'])} edges "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
