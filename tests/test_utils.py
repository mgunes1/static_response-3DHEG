"""Tests for src/utils.py."""
import numpy as np
import pytest

from src.utils import (
    box_length,
    coulomb_potential,
    density_from_rs,
    fermi_energy,
    fermi_wavevector,
    q_vectors_along_axis,
    unique_q_magnitudes,
)


class TestDensityFromRs:
    def test_known_value_rs1(self):
        # n = 3/(4π) for rs=1
        expected = 3.0 / (4.0 * np.pi)
        assert np.isclose(density_from_rs(1.0), expected)

    def test_scaling(self):
        # n ∝ rs^{-3}
        assert np.isclose(density_from_rs(2.0), density_from_rs(1.0) / 8.0)

    def test_positive(self):
        for rs in [0.5, 1.0, 2.0, 5.0, 10.0]:
            assert density_from_rs(rs) > 0.0


class TestFermiWavevector:
    def test_rs1(self):
        # kF = (9π/4)^{1/3} for rs=1
        expected = (9.0 * np.pi / 4.0) ** (1.0 / 3.0)
        assert np.isclose(fermi_wavevector(1.0), expected)

    def test_scaling(self):
        # kF ∝ 1/rs
        kf1 = fermi_wavevector(1.0)
        kf2 = fermi_wavevector(2.0)
        assert np.isclose(kf2, kf1 / 2.0)

    def test_positive(self):
        for rs in [1.0, 2.0, 5.0]:
            assert fermi_wavevector(rs) > 0.0


class TestFermiEnergy:
    def test_relation_to_kf(self):
        for rs in [1.0, 2.0, 5.0]:
            kf = fermi_wavevector(rs)
            assert np.isclose(fermi_energy(rs), 0.5 * kf**2)


class TestBoxLength:
    def test_volume_consistency(self):
        rs, N = 2.0, 14
        L = box_length(rs, N)
        n_from_box = N / L**3
        assert np.isclose(n_from_box, density_from_rs(rs), rtol=1e-10)

    def test_scaling_with_N(self):
        # L ∝ N^{1/3} at fixed rs
        rs = 2.0
        L14 = box_length(rs, 14)
        L112 = box_length(rs, 112)
        assert np.isclose(L112 / L14, (112 / 14) ** (1.0 / 3.0))


class TestCoulombPotential:
    def test_value(self):
        q = 1.0
        assert np.isclose(coulomb_potential(q), 4.0 * np.pi)

    def test_scaling(self):
        assert np.isclose(coulomb_potential(2.0), coulomb_potential(1.0) / 4.0)

    def test_array_input(self):
        q = np.array([1.0, 2.0, 3.0])
        v = coulomb_potential(q)
        assert v.shape == (3,)
        assert np.allclose(v, 4.0 * np.pi / q**2)


class TestQVectorsAlongAxis:
    def test_shape(self):
        q_vecs, q_mags = q_vectors_along_axis(2.0, 14, n_multiples=4)
        assert q_vecs.shape == (4, 3)
        assert q_mags.shape == (4,)

    def test_direction(self):
        q_vecs, _ = q_vectors_along_axis(2.0, 14, n_multiples=3, axis=0)
        # y and z components should be zero
        assert np.all(q_vecs[:, 1] == 0.0)
        assert np.all(q_vecs[:, 2] == 0.0)

    def test_spacing(self):
        rs, N = 2.0, 14
        q_vecs, q_mags = q_vectors_along_axis(rs, N, n_multiples=3)
        L = box_length(rs, N)
        q_unit = 2.0 * np.pi / L
        expected = np.array([q_unit, 2 * q_unit, 3 * q_unit])
        assert np.allclose(q_mags, expected)


class TestUniqueQMagnitudes:
    def test_minimum_q(self):
        rs, N = 2.0, 14
        q_mags, n_sq = unique_q_magnitudes(rs, N, n_shells=1)
        L = box_length(rs, N)
        q_unit = 2.0 * np.pi / L
        # Only |n|²=1 shell
        assert len(q_mags) == 1
        assert np.isclose(q_mags[0], q_unit)
        assert n_sq[0] == 1

    def test_sorted(self):
        q_mags, _ = unique_q_magnitudes(2.0, 14, n_shells=3)
        assert np.all(np.diff(q_mags) > 0)
