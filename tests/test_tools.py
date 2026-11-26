import tools


def test_fertilizer_plan_wheat():
    r = tools.fertilizer_plan("Wheat", 2)
    assert r["dap"]["kg"] > 0 and r["urea"]["kg"] > 0
    assert r["nutrient_target_kg_per_acre"]["N"] == 49


def test_fertilizer_bad_input():
    assert "error" in tools.fertilizer_plan("banana", 2)
    assert "error" in tools.fertilizer_plan("wheat", 0)


def test_sowing_window_shape():
    r = tools.sowing_window("cotton")
    assert r["crop"] == "cotton" and r["windows"]


def test_advisory_rain():
    notes = tools._advisory([{"date": "2026-01-01", "max_c": 20, "min_c": 8, "rain_mm": 5}])
    assert "avoid spraying" in notes[0]


def test_rag_keyword_fallback():
    import rag
    hits = rag.search("wheat yellow rust", top_k=2)
    assert hits and "wheat" in hits[0]["source"]
