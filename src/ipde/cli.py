"""Command-line interface for IPDE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .extractor import ExtractOptions, ExtractionError, extract_file, inspect_file


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ipde",
        description="Extract HEIF depth and auxiliary planes without normalization or display transforms.",
    )
    parser.add_argument("sources", nargs="+", type=Path, help="HEIC/HEIF input file(s)")
    parser.add_argument("--inspect", action="store_true", help="inventory assets without writing output files")
    parser.add_argument("--json", action="store_true", help="emit one JSON object per source")
    parser.add_argument("--output-dir", type=Path, help="output directory (default: each source directory)")
    parser.add_argument("--overwrite", action="store_true", help="replace colliding outputs after verification")
    parser.add_argument("--no-npy", action="store_true", help="omit exact NumPy array companions")
    parser.add_argument("--version", action="version", version=f"IPDE {__version__}")
    return parser


def _human_report(report: dict, inspected: bool) -> str:
    source = report["source"]
    lines = [
        f"{'Inspected' if inspected else 'Extracted'}: {source['path']}",
        f"Container: {source['mimetype']} | top-level images: {len(source['top_level_images'])}",
        f"Auxiliary planes: {report['asset_count']}",
    ]
    for asset in report["assets"]:
        shape = f"{asset['width']}x{asset['height']}"
        if asset["channels"] > 1:
            shape += f"x{asset['channels']}ch"
        lines.append(
            f"  {asset['semantic_name']}: {shape} {asset['dtype_name']} "
            f"(source {asset['source_bit_depth']}-bit, min={asset['minimum']}, max={asset['maximum']})"
        )
        for output in asset.get("outputs", []):
            lines.append(f"    {output['role']}: {output['path']}")
    for warning in report.get("warnings", []):
        lines.append(f"Warning: {warning}")
    if not inspected:
        lines.append(f"Manifest: {report['manifest_path']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    failed = False
    for source in args.sources:
        try:
            if args.inspect:
                report = inspect_file(source)
            else:
                report = extract_file(
                    source,
                    ExtractOptions(
                        output_dir=args.output_dir,
                        write_npy=not args.no_npy,
                        overwrite=args.overwrite,
                    ),
                )
            if args.json:
                print(json.dumps(report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False))
            else:
                print(_human_report(report, args.inspect))
        except (ExtractionError, OSError, ValueError) as exc:
            failed = True
            error = {"source": str(source.expanduser().resolve()), "error": str(exc)}
            if args.json:
                print(json.dumps(error, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            else:
                print(f"Error: {error['source']}: {error['error']}", file=sys.stderr)
    return 1 if failed else 0
