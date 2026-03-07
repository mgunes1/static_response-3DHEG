"""Tests for src/data_io.py."""
import os
import tempfile

import numpy as np
import pytest

from src.data_io import get_output_path, load_energy_matrix, save_energy_matrix


def _sample_data():
    n_q, n_lam = 3, 5
    return {
        "energies": np.random.default_rng(0).standard_normal((n_q, n_lam)),
        "q_vectors": np.random.default_rng(1).standard_normal((n_q, 3)),
        "q_magnitudes": np.array([0.8, 1.2, 1.8]),
        "lambdas": np.array([0.0, 0.05, 0.1, 0.15, 0.2]),
        "rs": 2.0,
        "N": 14,
        "energy_errors": np.full((n_q, n_lam), 5e-4),
        "metadata": {"method": "DMC", "stat_error": 5e-4},
    }


class TestSaveLoad:
    def test_roundtrip_basic(self):
        d = _sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            save_energy_matrix(
                path,
                d["energies"],
                d["q_vectors"],
                d["q_magnitudes"],
                d["lambdas"],
                d["rs"],
                d["N"],
            )
            loaded = load_energy_matrix(path)

        assert np.allclose(loaded["energies"], d["energies"])
        assert np.allclose(loaded["q_vectors"], d["q_vectors"])
        assert np.allclose(loaded["q_magnitudes"], d["q_magnitudes"])
        assert np.allclose(loaded["lambdas"], d["lambdas"])
        assert loaded["rs"] == d["rs"]
        assert loaded["N"] == d["N"]

    def test_roundtrip_with_errors(self):
        d = _sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            save_energy_matrix(
                path,
                d["energies"],
                d["q_vectors"],
                d["q_magnitudes"],
                d["lambdas"],
                d["rs"],
                d["N"],
                energy_errors=d["energy_errors"],
            )
            loaded = load_energy_matrix(path)

        assert "energy_errors" in loaded
        assert np.allclose(loaded["energy_errors"], d["energy_errors"])

    def test_roundtrip_with_metadata(self):
        d = _sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            save_energy_matrix(
                path,
                d["energies"],
                d["q_vectors"],
                d["q_magnitudes"],
                d["lambdas"],
                d["rs"],
                d["N"],
                metadata=d["metadata"],
            )
            loaded = load_energy_matrix(path)

        assert "metadata" in loaded
        assert loaded["metadata"]["stat_error"] == pytest.approx(5e-4)

    def test_load_auto_appends_npz(self):
        d = _sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test")
            save_energy_matrix(
                path, d["energies"], d["q_vectors"], d["q_magnitudes"],
                d["lambdas"], d["rs"], d["N"],
            )
            # Load without extension
            loaded = load_energy_matrix(path)
            # Load with extension
            loaded2 = load_energy_matrix(path + ".npz")

        assert np.allclose(loaded["energies"], loaded2["energies"])

    def test_creates_parent_directories(self):
        d = _sample_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "a", "b", "c", "test")
            save_energy_matrix(
                path, d["energies"], d["q_vectors"], d["q_magnitudes"],
                d["lambdas"], d["rs"], d["N"],
            )
            assert os.path.exists(path + ".npz")

    def test_load_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                load_energy_matrix(os.path.join(tmpdir, "nonexistent"))


class TestGetOutputPath:
    def test_format(self):
        p = get_output_path("output", 2.0, 14)
        assert p == os.path.join("output", "rs2.0", "N14", "energies")

    def test_suffix(self):
        p = get_output_path("output", 2.0, 14, suffix="_backflow")
        assert p.endswith("energies_backflow")

    def test_float_formatting(self):
        # rs=10 should format as 10.0
        p = get_output_path("output", 10.0, 54)
        assert "rs10.0" in p
        assert "N54" in p
