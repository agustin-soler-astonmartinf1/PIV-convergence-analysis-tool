from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import extract_uvw


class ExtractUVWTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_dat = REPO_ROOT / "Cam1_0011.dat"
        cls.sample_vectors = extract_uvw.load_valid_vectors(cls.sample_dat)

    def test_load_valid_vectors_retains_visualizer_fields(self):
        self.assertTrue(self.sample_vectors)
        first = self.sample_vectors[0]

        self.assertEqual(
            set(first),
            {"x", "y", "z", "u", "v", "w", "velmag", "flag"},
        )
        self.assertEqual(first["flag"], 3)

    def test_extract_uvw_from_file_returns_exact_match(self):
        first = self.sample_vectors[0]

        result = extract_uvw.extract_uvw_from_file(
            self.sample_dat,
            target_x=first["x"],
            target_y=first["y"],
            tol=1e-4,
            fallback="none",
            max_distance=None,
        )

        self.assertEqual(result["status"], "exact")
        self.assertEqual(result["vector"][0], first["x"])
        self.assertEqual(result["vector"][1], first["y"])
        self.assertEqual(result["record"]["velmag"], first["velmag"])

    def test_resolve_preview_source_supports_folder_and_file_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            input_dir.mkdir()

            second_file = input_dir / "02_second.dat"
            first_file = input_dir / "01_first.dat"
            shutil.copyfile(self.sample_dat, second_file)
            shutil.copyfile(self.sample_dat, first_file)

            folder_result = extract_uvw.resolve_preview_source(input_dir)
            file_result = extract_uvw.resolve_preview_source(second_file)

            self.assertEqual(folder_result["folder"], input_dir)
            self.assertEqual(folder_result["preview_file"], first_file)
            self.assertEqual(
                [path.name for path in folder_result["dat_files"]],
                ["01_first.dat", "02_second.dat"],
            )

            self.assertEqual(file_result["folder"], input_dir)
            self.assertEqual(file_result["preview_file"], second_file)
            self.assertEqual(
                [path.name for path in file_result["dat_files"]],
                ["01_first.dat", "02_second.dat"],
            )

    def test_run_extraction_writes_expected_csv(self):
        first = self.sample_vectors[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            input_dir.mkdir()
            shutil.copyfile(self.sample_dat, input_dir / self.sample_dat.name)

            output_csv = temp_path / "result.csv"
            summary = extract_uvw.run_extraction(
                input_dir,
                target_x=first["x"],
                target_y=first["y"],
                output_csv=output_csv,
            )

            self.assertEqual(summary["exact_count"], 1)
            self.assertEqual(summary["nearest_count"], 0)
            self.assertEqual(summary["missing_count"], 0)
            self.assertTrue(output_csv.exists())

            with output_csv.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(rows[0], extract_uvw.CSV_HEADER)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][0], self.sample_dat.name)
        self.assertEqual(rows[1][3], "exact")

    def test_run_extraction_processes_dat_files_in_alphabetical_order(self):
        first = self.sample_vectors[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            input_dir.mkdir()

            for file_name in ["03_third.dat", "01_first.dat", "02_second.dat"]:
                shutil.copyfile(self.sample_dat, input_dir / file_name)

            output_csv = temp_path / "result.csv"
            summary = extract_uvw.run_extraction(
                input_dir,
                target_x=first["x"],
                target_y=first["y"],
                output_csv=output_csv,
            )

            self.assertEqual(
                [path.name for path in summary["dat_files"]],
                ["01_first.dat", "02_second.dat", "03_third.dat"],
            )
            self.assertEqual(
                [row[0] for row in summary["rows"]],
                ["01_first.dat", "02_second.dat", "03_third.dat"],
            )


if __name__ == "__main__":
    unittest.main()