#!/usr/bin/env python3
"""merge_routes.py — Merge individual route XML files into routes_codriving.xml.

Usage:
    python merge_routes.py
    python merge_routes.py --output routes_codriving.xml
"""

import os, sys, glob
from xml.etree import ElementTree as ET

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "simulation", "leaderboard", "data")
INTERDRIVE_DIR = os.path.join(DATA_DIR, "Interdrive")
OUTPUT_FILE = os.path.join(DATA_DIR, "routes_codriving.xml")


def merge_routes(output=OUTPUT_FILE):
    xml_files = sorted(glob.glob(os.path.join(INTERDRIVE_DIR, "**", "*.xml"), recursive=True))
    print(f"Found {len(xml_files)} route XML files")

    root = ET.Element("routes")
    total_routes = 0
    towns = set()

    for fpath in xml_files:
        try:
            tree = ET.parse(fpath)
            for route in tree.getroot().findall("route"):
                town = route.get("town", "unknown")
                towns.add(town)
                route_id = route.get("id", "0")
                route.set("id", str(total_routes))

                # Add weather and scenario tags if not present
                if route.find("weather") is None:
                    ET.SubElement(route, "weather").set("cloudiness", "0")
                    ET.SubElement(route, "weather").set("precipitation", "0")
                    ET.SubElement(route, "weather").set("wind_intensity", "0")
                    ET.SubElement(route, "weather").set("sun_altitude_angle", "70")

                root.append(route)
                total_routes += 1
        except ET.ParseError as e:
            print(f"  [SKIP] {os.path.basename(fpath)}: {e}")

    tree = ET.ElementTree(root)
    tree.write(output, encoding="utf-8", xml_declaration=True)
    print(f"Merged {total_routes} routes from towns: {sorted(towns)}")
    print(f"Written to: {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    merge_routes(args.output)
