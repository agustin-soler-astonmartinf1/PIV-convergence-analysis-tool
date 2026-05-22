#!/usr/bin/env python3
"""
Extract u, v, w values at a specified x, y point from multiple PIV .dat files.

Input:
    - Folder containing .dat files
    - x coordinate
    - y coordinate
    - Output CSV path

Output CSV columns:
    filename, u, v, w

Example:
    python extract_uvw.py "C:/data/Cam1" -521.3 141.788 output.csv

With a larger coordinate tolerance:
    python extract_uvw.py "C:/data/Cam1" -6.3 -158.212 output.csv --tol 1e-3
"""

import argparse
import csv
import math
from pathlib import Path


def parse_numeric_line(line):
    """
    Try to parse a whitespace-separated numeric data line.

    Expected .dat row format:
        x y z u v w ...
    """
    parts = line.strip().split()

    if len(parts) < 6:
        return None

    try:
        values = [float(p) for p in parts[:6]]
    except ValueError:
        return None

    return values  # x, y, z, u, v, w


def extract_uvw_from_file(file_path, target_x, target_y, tol):
    """
    Return (u, v, w) for the row matching target_x and target_y.

    Returns None if no matching point is found.
    """
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            values = parse_numeric_line(line)

            if values is None:
                continue

            x, y, _z, u, v, w = values

            if (
                math.isclose(x, target_x, abs_tol=tol, rel_tol=0.0)
                and math.isclose(y, target_y, abs_tol=tol, rel_tol=0.0)
            ):
                return u, v, w

    return None


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
        help="Target x coordinate."
    )

    parser.add_argument(
        "y",
        type=float,
        help="Target y coordinate."
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

    args = parser.parse_args()

    if not args.folder.is_dir():
        raise NotADirectoryError(f"Input folder does not exist: {args.folder}")

    dat_files = sorted(args.folder.glob("*.dat"), key=lambda p: p.name)

    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in folder: {args.folder}")

    with args.output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "u", "v", "w"])

        for dat_file in dat_files:
            uvw = extract_uvw_from_file(
                dat_file,
                target_x=args.x,
                target_y=args.y,
                tol=args.tol,
            )

            if uvw is None:
                if args.missing == "zero":
                    row = [dat_file.name, 0.0, 0.0, 0.0]
                else:
                    row = [dat_file.name, "", "", ""]
            else:
                u, v, w = uvw
                row = [dat_file.name, u, v, w]

            writer.writerow(row)

    print(f"Processed {len(dat_files)} .dat files.")
    print(f"Output written to: {args.output_csv}")


if __name__ == "__main__":
    main()