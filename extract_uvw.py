#!/usr/bin/env python3
"""
Extract u, v, w values at a specified x, y point from multiple PIV .dat files.

Input:
    - Folder containing .dat files
    - x coordinate
    - y coordinate
    - Output CSV path

Output CSV columns:
    filename, requested_x, requested_y, match_status, match_distance,
    vector_x, vector_y, u, v, w

Example:
    python extract_uvw.py "C:/data/Cam1" -521.3 141.788 output.csv

With a larger coordinate tolerance:
    python extract_uvw.py "C:/data/Cam1" -6.3 -158.212 output.csv --tol 1e-3

To allow nearest-neighbor fallback when no exact point is found:
    python extract_uvw.py "C:/data/Cam1" -6.3 -158.212 output.csv --fallback nearest
"""

import argparse
import csv
import math
from pathlib import Path


def parse_numeric_line(line):
    """
    Try to parse a whitespace-separated numeric data line.

    Expected .dat row format:
        x y z u v w velmag flag ...

    Only rows marked as ValidData (flag=3) are returned.
    """
    parts = line.strip().split()

    if len(parts) < 8:
        return None

    try:
        values = [float(p) for p in parts[:8]]
    except ValueError:
        return None

    x, y, z, u, v, w, _velmag, flag = values

    if flag != 3.0:
        return None

    return x, y, z, u, v, w


def extract_uvw_from_file(file_path, target_x, target_y, tol, fallback, max_distance):
    """
    Return the selected valid vector metadata for target_x and target_y.

    Exact matches are returned immediately.
    Nearest-neighbor fallback is only used when requested explicitly.

    The returned dict always contains:
        status: exact, nearest, no_match, nearest_too_far, or no_valid_vectors
        distance: Euclidean distance from the requested point, if available
        vector: (x, y, u, v, w) for exact or nearest matches, otherwise None
    """
    closest_vector = None
    closest_distance = None
    saw_valid_vector = False

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = parse_numeric_line(line)

            if values is None:
                continue

            saw_valid_vector = True
            x, y, _z, u, v, w = values
            distance = math.hypot(x - target_x, y - target_y)

            if (
                math.isclose(x, target_x, abs_tol=tol, rel_tol=0.0)
                and math.isclose(y, target_y, abs_tol=tol, rel_tol=0.0)
            ):
                return {
                    "status": "exact",
                    "distance": distance,
                    "vector": (x, y, u, v, w),
                }

            if closest_distance is None or distance < closest_distance:
                closest_distance = distance
                closest_vector = (x, y, u, v, w)

    if not saw_valid_vector:
        return {
            "status": "no_valid_vectors",
            "distance": None,
            "vector": None,
        }

    if fallback == "nearest" and closest_vector is not None:
        if max_distance is not None and closest_distance > max_distance:
            return {
                "status": "nearest_too_far",
                "distance": closest_distance,
                "vector": None,
            }

        return {
            "status": "nearest",
            "distance": closest_distance,
            "vector": closest_vector,
        }

    return {
        "status": "no_match",
        "distance": closest_distance,
        "vector": None,
    }


def get_valid_coordinate_bounds(file_path):
    """Return min/max physical x and y for valid vectors in one .dat file."""
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = parse_numeric_line(line)

            if values is None:
                continue

            x, y, _z, _u, _v, _w = values

            if min_x is None:
                min_x = max_x = x
                min_y = max_y = y
                continue

            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    if min_x is None:
        return None

    return min_x, max_x, min_y, max_y


def main():
    parser = argparse.ArgumentParser(
        description="Extract u, v, w at a given x, y point from all .dat files in a folder."
    )

    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing the .dat files."
    )

    parser.add_argument(
        "x",
        type=float,
        help="Target x coordinate in the .dat physical coordinate system."
    )

    parser.add_argument(
        "y",
        type=float,
        help="Target y coordinate in the .dat physical coordinate system."
    )

    parser.add_argument(
        "output_csv",
        type=Path,
        help="Output CSV filename."
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Absolute tolerance for matching x and y coordinates. Default: 1e-4"
    )

    parser.add_argument(
        "--missing",
        choices=["blank", "zero"],
        default="blank",
        help="What to write if the point is not found in a file. Default: blank"
    )

    parser.add_argument(
        "--fallback",
        choices=["none", "nearest"],
        default="none",
        help="How to handle files without an exact x/y match. Default: none"
    )

    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum Euclidean distance allowed for --fallback nearest. Default: no limit"
    )

    args = parser.parse_args()

    if args.max_distance is not None and args.max_distance < 0:
        raise ValueError("--max-distance must be non-negative")

    if not args.folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {args.folder}")

    dat_files = sorted(args.folder.glob("*.dat"), key=lambda p: p.name)

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in folder: {args.folder}")

    exact_count = 0
    nearest_count = 0
    missing_count = 0

    with args.output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "filename",
                "requested_x",
                "requested_y",
                "match_status",
                "match_distance",
                "vector_x",
                "vector_y",
                "u",
                "v",
                "w",
            ]
        )

        for dat_file in dat_files:
            result = extract_uvw_from_file(
                dat_file,
                target_x=args.x,
                target_y=args.y,
                tol=args.tol,
                fallback=args.fallback,
                max_distance=args.max_distance,
            )

            status = result["status"]
            distance = result["distance"]

            if status == "exact":
                exact_count += 1
            elif status == "nearest":
                nearest_count += 1
            else:
                missing_count += 1

            row = [
                dat_file.name,
                args.x,
                args.y,
                status,
                "" if distance is None else distance,
            ]

            if result["vector"] is None:
                if args.missing == "zero":
                    row.extend(["", "", 0.0, 0.0, 0.0])
                else:
                    row.extend(["", "", "", "", ""])
            else:
                vector_x, vector_y, u, v, w = result["vector"]
                row.extend([vector_x, vector_y, u, v, w])

            writer.writerow(row)

    print(f"Processed {len(dat_files)} .dat files.")
    print(f"Exact matches: {exact_count}")
    print(f"Nearest matches: {nearest_count}")
    print(f"Missing matches: {missing_count}")
    if missing_count:
        print(
            "No exact point was found for some files. "
            "Check whether the requested x/y are in the .dat physical coordinate system."
        )
        if missing_count == len(dat_files):
            bounds = get_valid_coordinate_bounds(dat_files[0])
            if bounds is not None:
                min_x, max_x, min_y, max_y = bounds
                print(
                    "Available valid coordinates in "
                    f"{dat_files[0].name} span x={min_x} to {max_x}, "
                    f"y={min_y} to {max_y}."
                )

            nearest_result = extract_uvw_from_file(
                dat_files[0],
                target_x=args.x,
                target_y=args.y,
                tol=args.tol,
                fallback="nearest",
                max_distance=None,
            )
            nearest_vector = nearest_result["vector"]
            if nearest_vector is not None:
                nearest_x, nearest_y, _u, _v, _w = nearest_vector
                print(
                    f"Closest valid point in {dat_files[0].name} is "
                    f"x={nearest_x}, y={nearest_y} "
                    f"(distance={nearest_result['distance']})."
                )
    print(f"Output written to: {args.output_csv}")


if __name__ == "__main__":
    main()