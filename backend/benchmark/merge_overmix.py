#!/usr/bin/env python3
"""
Merges Overmix per-test artifacts (overmix_stitch.png, overmix_variant.json)
into the latest benchmark report without requiring a full pipeline re-run.
(Implementation of ASP Roadmap Phase 0.3)
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

# Load bench_anime_stitch module
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import bench_anime_stitch as bas
except ImportError:
    print("Error: Could not import bench_anime_stitch.py")
    sys.exit(1)

import cv2

def main():
    parser = argparse.ArgumentParser(description="Merge Overmix artifacts into the latest benchmark JSON/Markdown report.")
    parser.add_argument(
        "--data-dir",
        default=os.path.expanduser("~/Downloads/Data/Dump"),
        metavar="DIR",
        help="Root data directory containing asp_testXX subdirectories",
    )
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    candidates = sorted(glob.glob(os.path.join(results_dir, "anime_stitch_*.json")))
    if not candidates:
        print(f"No anime_stitch_*.json found in {results_dir}")
        return

    latest_json = candidates[-1]
    print(f"Loading baseline from {latest_json}...")
    try:
        with open(latest_json, "r") as fh:
            doc = json.load(fh)
    except Exception as e:
        print(f"Failed to load {latest_json}: {e}")
        return
        
    results = doc.get("datasets", [])
    if not results:
        print("No datasets found in the JSON document.")
        return
        
    updated_count = 0
    for r in results:
        dataset_name = r.get("name")
        if not dataset_name:
            continue
            
        dataset_dir = os.path.join(args.data_dir, dataset_name)
        output_dir = os.path.join(dataset_dir, "output")
        overmix_path = os.path.join(output_dir, "overmix_stitch.png")
        variant_path = os.path.join(output_dir, "overmix_variant.json")
        
        if os.path.exists(overmix_path):
            img = cv2.imread(overmix_path)
            if img is not None:
                # Generate metrics exactly as bench_anime_stitch does
                metrics = bas._compute_all_metrics(img)
                
                # Also include data from variant.json if available
                if os.path.exists(variant_path):
                    try:
                        with open(variant_path, "r") as fh:
                            variant_data = json.load(fh)
                            # E.g., add smart variant time to metrics
                            if "smart" in variant_data and "wall_sec" in variant_data["smart"]:
                                metrics["overmix_wall_sec"] = variant_data["smart"]["wall_sec"]
                    except Exception:
                        pass
                        
                r["metrics_overmix"] = metrics
                r["overmix_path"] = overmix_path
                updated_count += 1
                print(f"Updated {dataset_name} with Overmix metrics.")
            else:
                print(f"Failed to read image {overmix_path}")
                
    if updated_count > 0:
        suite_start_time = time.perf_counter()
        print("\nGenerating new JSON results...")
        new_json_path = bas.generate_json_results(results, suite_start_time)
        print(f"Saved: {new_json_path}")
        
        print("Generating new Markdown report...")
        report_path = bas.generate_report(results, os.path.join(args.data_dir, "output"))
        print(f"Saved: {report_path}")
    else:
        print("\nNo overmix_stitch.png images found to merge.")

if __name__ == "__main__":
    main()
