import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path("/home/yektefai/PhylaFlow")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.full_sanity_fixedpair_20260401.make_singlepath_parity_case import (
    _build_anchor_payloads,
)


def _load_tree(json_path: str) -> str:
    payload = json.loads(Path(json_path).read_text())
    return str(payload["tree"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, default=None)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--full-path-anchor-count", type=int, default=4)
    args = parser.parse_args()

    manifest = json.loads(args.source_manifest.read_text())
    cases = list(manifest.get("cases", []))
    if not cases:
        raise RuntimeError("Source manifest has no cases.")

    analysis_dir = ROOT / "analysis/full_sanity_fixedpair_20260401"
    configs_dir = ROOT / "configs"
    output_anchor_json = analysis_dir / f"{args.output_name}_velocity_anchors.json"
    output_manifest_json = analysis_dir / f"{args.output_name}_manifest.json"
    output_config_yaml = configs_dir / f"{args.output_name}.yaml"

    combined_anchors = []
    case_summaries = []
    total_boundary_paths = 0
    for case in cases:
        case_name = str(case["case_name"])
        start_tree = _load_tree(case["start_json"])
        target_tree = _load_tree(case["target_json"])
        anchor_payload = _build_anchor_payloads(
            start_tree,
            target_tree,
            bank_group_key=case_name,
            o0_count=4,
            a1_count=4,
            o2_count=4,
            full_path_count=int(args.full_path_anchor_count),
        )
        combined_anchors.extend(anchor_payload["anchors"])
        total_boundary_paths += int(anchor_payload["boundary_path_count"])
        case_summaries.append(
            {
                "case_name": case_name,
                "boundary_path_count": int(anchor_payload["boundary_path_count"]),
                "anchor_count": int(len(anchor_payload["anchors"])),
                "start_json": str(case["start_json"]),
                "target_json": str(case["target_json"]),
                "topology_key": case.get("topology_key"),
                "topology_count": case.get("topology_count"),
                "topology_probability": case.get("topology_probability"),
            }
        )

    output_anchor_json.write_text(json.dumps(combined_anchors, indent=2))

    payload = {
        "source_manifest": str(args.source_manifest),
        "source_config": None if args.source_config is None else str(args.source_config),
        "output_name": str(args.output_name),
        "num_cases": int(len(cases)),
        "full_path_anchor_count": int(args.full_path_anchor_count),
        "total_anchor_count": int(len(combined_anchors)),
        "mean_boundary_path_count": float(total_boundary_paths) / float(len(cases)),
        "mean_anchor_count_per_case": float(len(combined_anchors)) / float(len(cases)),
        "combined_anchors_json": str(output_anchor_json),
        "cases": case_summaries,
    }

    if args.source_config is not None:
        config = yaml.safe_load(args.source_config.read_text())
        config["trainer"]["checkpoint_dir"] = (
            f"./checkpoints/full_sanity_fixedpair_20260401/{args.output_name}"
        )
        config["trainer"]["sample_metrics_trace_path"] = str(
            analysis_dir / f"{args.output_name}_metrics.jsonl"
        )
        config["data"]["overfit_full_path_control_extra_velocity_samples_json_path"] = str(
            output_anchor_json
        )
        output_config_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
        payload["config_yaml"] = str(output_config_yaml)

    output_manifest_json.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
