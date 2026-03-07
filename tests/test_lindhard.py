"""Tests for src/lindhard.py."""
import numpy as np
import pytest

from src.lindhard import chi0, chi_rpa, lindhard_function, local_field_factor
from src.utils import fermi_wavevector


class TestLindhard:
    def test_x_equals_zero_limit(self):
        # L(0) → 1
        assert np.isclose(lindhard_function(1e-10), 1.0, atol=1e-6)

    def test_x_equals_one(self):
        # L(1) = 1/2 exactly
        assert np.isclose(lindhard_function(1.0), 0.5)

    def test_x_equals_two(self):
        # Manual: L(2) = 0.5 + (1-4)/(8) * ln(3) = 0.5 - (3/8)*ln3
        expected = 0.5 + (1.0 - 4.0) / (4.0 * 2.0) * np.log(3.0 / 1.0)
        assert np.isclose(lindhard_function(2.0), expected)

    def test_array_input(self):
        x = np.array([0.5, 1.0, 1.5, 2.0])
        result = lindhard_function(x)
        assert result.shape == (4,)

    def test_scalar_input_returns_scalar(self):
        result = lindhard_function(0.5)
        assert np.ndim(result) == 0

    def test_large_x_decay(self):
        # L(x) → 0 as x → ∞; for x=10 should be small
        assert lindhard_function(10.0) < 0.1

    def test_positive_range(self):
        # L(x) ≥ 0 for x ≥ 0
        x = np.linspace(1e-3, 5.0, 100)
        assert np.all(lindhard_function(x) >= 0.0)


class TestChi0:
    def test_negative_definite(self):
        # χ₀(q) < 0 for all q > 0
        rs = 2.0
        q = np.array([0.5, 1.0, 2.0, 5.0])
        assert np.all(chi0(q, rs) < 0.0)

    def test_small_q_limit(self):
        # χ₀(q→0) → −kF/π²
        rs = 2.0
        kf = fermi_wavevector(rs)
        expected = -kf / np.pi**2
        assert np.isclose(chi0(1e-6, rs), expected, rtol=1e-4)

    def test_scaling_with_rs(self):
        # kF ∝ 1/rs  →  χ₀(q,rs) = −(1/(π² rs)) L(...)
        # At q→0: χ₀ = −kF/π² ∝ 1/rs
        c1 = chi0(1e-6, 1.0)
        c2 = chi0(1e-6, 2.0)
        assert np.isclose(c1 / c2, 2.0, rtol=1e-3)


class TestChiRPA:
    def test_negative(self):
        # χ_RPA < 0 for all q
        rs = 2.0
        q = np.array([0.3, 0.8, 1.5])
        assert np.all(chi_rpa(q, rs) < 0.0)

    def test_rpa_screened_relative_to_chi0(self):
        # Coulomb screening reduces |χ_RPA| below |χ₀| at small q
        # (the denominator 1 − v·χ₀ > 1 because v>0, χ₀<0)
        rs = 2.0
        q = 0.5
        assert abs(chi_rpa(q, rs)) < abs(chi0(q, rs))

    def test_large_q_approaches_chi0(self):
        # At q → ∞, v(q)→0 so χ_RPA → χ₀
        rs = 2.0
        q = 100.0
        assert np.isclose(chi_rpa(q, rs), chi0(q, rs), rtol=1e-3)


class TestLocalFieldFactor:
    def test_rpa_gives_zero_G(self):
        # If we feed χ_RPA as "DMC" result, G(q) should be ≈ 0
        rs = 2.0
        q = np.array([0.5, 1.0, 2.0])
        chi_dmc = chi_rpa(q, rs)
        G = local_field_factor(q, chi_dmc, rs)
        assert np.allclose(G, 0.0, atol=1e-10)

    def test_chi0_input_gives_G_equals_1(self):
        # Feeding χ₀ as the "DMC" result: G = 1 + (1/v)(1/χ₀ − 1/χ₀) = 1
        rs = 2.0
        q = 1.0
        G = local_field_factor(q, chi0(q, rs), rs)
        assert np.isclose(G, 1.0, rtol=1e-8)

    def test_output_shape_matches_input(self):
        rs = 2.0
        q = np.linspace(0.5, 3.0, 10)
        chi_dmc = chi_rpa(q, rs) * 1.1  # arbitrary modification
        G = local_field_factor(q, chi_dmc, rs)
        assert G.shape == (10,)
