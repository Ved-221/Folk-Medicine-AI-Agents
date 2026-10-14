"""Unit tests for Agent 2 age banding and deterministic dosage lookup."""

import pytest
from src.agent2.age_band import age_to_band, lookup_dosage


def test_age_to_band_boundaries():
    assert age_to_band(1) == "infant_under_3"
    assert age_to_band(2) == "infant_under_3"
    assert age_to_band(3) == "child_3_12_years"
    assert age_to_band(11) == "child_3_12_years"
    assert age_to_band(12) == "adolescent_12_17_years"
    assert age_to_band(17) == "adolescent_12_17_years"
    assert age_to_band(18) == "adult_18_60_years"
    assert age_to_band(35) == "adult_18_60_years"
    assert age_to_band(60) == "adult_18_60_years"
    assert age_to_band(61) == "elderly_60_plus_years"
    assert age_to_band(90) == "elderly_60_plus_years"
    assert age_to_band(120) == "elderly_60_plus_years"


def test_age_to_band_invalid_ages():
    with pytest.raises(ValueError):
        age_to_band(0)
    with pytest.raises(ValueError):
        age_to_band(-10)
    with pytest.raises(ValueError):
        age_to_band(121)
    with pytest.raises(ValueError):
        age_to_band("30")  # type: ignore


def test_lookup_dosage_found_in_source():
    row = {
        "dosage": {"amount": "500mg", "frequency": "twice daily"},
        "age_suitability": {
            "dosage_adjustments_by_age": {
                "adult_18_60_years": "500mg twice daily",
                "child_3_12_years": "250mg once daily",
            }
        },
    }
    dose_adult, origin_adult = lookup_dosage(row, "adult_18_60_years")
    assert dose_adult == "500mg twice daily"
    assert origin_adult == "from_source"

    dose_child, origin_child = lookup_dosage(row, "child_3_12_years")
    assert dose_child == "250mg once daily (twice daily)" or "250mg" in dose_child
    assert origin_child == "from_source"


def test_lookup_dosage_missing_band():
    row = {
        "dosage": {"amount": "500mg"},
        "age_suitability": {"dosage_adjustments_by_age": {}},
    }
    dose, origin = lookup_dosage(row, "child_3_12_years")
    assert "Not specified in source" in dose
    assert origin == "from_source"


def test_lookup_dosage_infant_boundary():
    row = {"dosage": {"amount": "500mg"}}
    dose, origin = lookup_dosage(row, "infant_under_3")
    assert "infants under 3" in dose
    assert origin == "from_source"
