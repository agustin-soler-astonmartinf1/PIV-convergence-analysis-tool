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


CSV_HEADER = [
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

    x, y, z, u, v, w, velmag, flag = values

    if flag != 3.0:
        return None

    return {
        "x": x,
        "y": y,
        "z": z,
        "u": u,
        "v": v,
        "w": w,
        "velmag": velmag,
        "flag": int(flag),
    }


def load_valid_vectors(file_path):
    """Load all valid vectors from one .dat file."""
    vectors = []

    with Path(file_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = parse_numeric_line(line)

            if values is not None:
                vectors.append(values)

    return vectors


def vector_record_to_tuple(record):
    """Convert a parsed vector record into the CSV/result tuple shape."""
    return (record["x"], record["y"], record["u"], record["v"], record["w"])


def extract_uvw_from_vectors(vectors, target_x, target_y, tol, fallback, max_distance):
    """Select the best matching vector from an in-memory valid-vector list."""
    closest_vector = None
    closest_distance = None

    for vector in vectors:
        distance = math.hypot(vector["x"] - target_x, vector["y"] - target_y)

        if (
            math.isclose(vector["x"], target_x, abs_tol=tol, rel_tol=0.0)
            and math.isclose(vector["y"], target_y, abs_tol=tol, rel_tol=0.0)
        ):
            return {
                "status": "exact",
                "distance": distance,
                "vector": vector_record_to_tuple(vector),
                "record": vector,
            }

        if closest_distance is None or distance < closest_distance:
            closest_distance = distance
            closest_vector = vector

    if not vectors:
        return {
            "status": "no_valid_vectors",
            "distance": None,
            "vector": None,
            "record": None,
        }

    if fallback == "nearest" and closest_vector is not None:
        if max_distance is not None and closest_distance > max_distance:
            return {
                "status": "nearest_too_far",
                "distance": closest_distance,
                "vector": None,
                "record": None,
            }

        return {
            "status": "nearest",
            "distance": closest_distance,
            "vector": vector_record_to_tuple(closest_vector),
            "record": closest_vector,
        }

    return {
        "status": "no_match",
        "distance": closest_distance,
        "vector": None,
        "record": None,
    }


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
    vectors = load_valid_vectors(file_path)
    return extract_uvw_from_vectors(
        vectors,
        target_x=target_x,
        target_y=target_y,
        tol=tol,
        fallback=fallback,
        max_distance=max_distance,
    )


def find_dat_files(folder):
    """Return sorted .dat files from a folder, validating the folder first."""
    folder = Path(folder)

    if not folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {folder}")

    dat_files = sorted(folder.glob("*.dat"), key=lambda p: p.name)

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in folder: {folder}")

    return dat_files


def resolve_preview_source(path):
    """Resolve a selected .dat file or folder into preview and batch paths."""
    path = Path(path)

    if path.is_dir():
        dat_files = find_dat_files(path)
        return {
            "folder": path,
            "preview_file": dat_files[0],
            "dat_files": dat_files,
        }

    if path.is_file():
        if path.suffix.lower() != ".dat":
            raise ValueError(f"Selected file is not a .dat file: {path}")

        dat_files = find_dat_files(path.parent)
        return {
            "folder": path.parent,
            "preview_file": path,
            "dat_files": dat_files,
        }

    if path.suffix.lower() == ".dat":
        raise FileNotFoundError(f"Selected .dat file does not exist: {path}")

    raise FileNotFoundError(f"Selected path does not exist: {path}")


def build_output_row(file_name, requested_x, requested_y, result, missing):
    """Build one CSV row from an extraction result."""
    distance = result["distance"]
    row = [
        file_name,
        requested_x,
        requested_y,
        result["status"],
        "" if distance is None else distance,
    ]

    if result["vector"] is None:
        if missing == "zero":
            row.extend(["", "", 0.0, 0.0, 0.0])
        else:
            row.extend(["", "", "", "", ""])

        return row

    vector_x, vector_y, u, v, w = result["vector"]
    row.extend([vector_x, vector_y, u, v, w])
    return row


def extract_uvw_from_folder(
    folder,
    target_x,
    target_y,
    tol=1e-4,
    missing="blank",
    fallback="none",
    max_distance=None,
):
    """Run extraction across every .dat file in a folder and return rows plus counts."""
    dat_files = find_dat_files(folder)
    rows = []
    exact_count = 0
    nearest_count = 0
    missing_count = 0

    for dat_file in dat_files:
        result = extract_uvw_from_file(
            dat_file,
            target_x=target_x,
            target_y=target_y,
            tol=tol,
            fallback=fallback,
            max_distance=max_distance,
        )

        status = result["status"]

        if status == "exact":
            exact_count += 1
        elif status == "nearest":
            nearest_count += 1
        else:
            missing_count += 1

        rows.append(
            build_output_row(
                dat_file.name,
                requested_x=target_x,
                requested_y=target_y,
                result=result,
                missing=missing,
            )
        )

    return {
        "folder": Path(folder),
        "dat_files": dat_files,
        "rows": rows,
        "exact_count": exact_count,
        "nearest_count": nearest_count,
        "missing_count": missing_count,
    }


def write_output_csv(output_csv, rows):
    """Write extraction rows to a CSV file."""
    output_csv = Path(output_csv)

    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    return output_csv


def run_extraction(
    folder,
    target_x,
    target_y,
    output_csv,
    tol=1e-4,
    missing="blank",
    fallback="none",
    max_distance=None,
):
    """Run the batch extraction flow and write the resulting CSV."""
    summary = extract_uvw_from_folder(
        folder,
        target_x=target_x,
        target_y=target_y,
        tol=tol,
        missing=missing,
        fallback=fallback,
        max_distance=max_distance,
    )
    summary["output_csv"] = write_output_csv(output_csv, summary["rows"])
    return summary


def get_valid_coordinate_bounds(file_path):
    """Return min/max physical x and y for valid vectors in one .dat file."""
    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for vector in load_valid_vectors(file_path):
        x = vector["x"]
        y = vector["y"]

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

    summary = run_extraction(
        args.folder,
        target_x=args.x,
        target_y=args.y,
        output_csv=args.output_csv,
        tol=args.tol,
        missing=args.missing,
        fallback=args.fallback,
        max_distance=args.max_distance,
    )
    dat_files = summary["dat_files"]
    exact_count = summary["exact_count"]
    nearest_count = summary["nearest_count"]
    missing_count = summary["missing_count"]

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
    print(f"Output written to: {summary['output_csv']}")


if __name__ == "__main__":
    main()