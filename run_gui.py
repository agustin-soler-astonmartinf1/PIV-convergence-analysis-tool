#!/usr/bin/env python3
"""Launch the Tkinter visualizer for the PIV convergence analysis tool."""

from __future__ import annotations

import sys
from pathlib import Path

import piv_visualizer


def main():
    initial_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    piv_visualizer.main(initial_folder)


if __name__ == "__main__":
    main()