def test_list_cyclones(client):
    response = client.get("/api/cyclones")
    assert response.status_code == 200
    storms = response.json()
    assert isinstance(storms, list)
    assert len(storms) >= 1
    ids = [s["id"] for s in storms]
    assert "LIVE-NIO-SATELLITE" in ids


def test_get_cyclone_live_stream(client):
    response = client.get("/api/cyclones/LIVE-NIO-SATELLITE")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "LIVE-NIO-SATELLITE"
    assert "NORTH INDIAN OCEAN" in data.get("basin", "").upper() or "NORTH INDIAN OCEAN" in data.get("name", "").upper()
    assert data["provenance"]["dataType"] == "LIVE SATELLITE STREAM"


def test_get_cyclone_amphan_authentic(client):
    response = client.get("/api/cyclones/IMD-2020-AMPHAN")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AMPHAN"
    assert data["windSpeed"] > 0
    assert data["provenance"]["dataType"] in ("VERIFIED HISTORICAL DATA", "VERIFIED OBSERVATIONAL DATA")


def test_get_cyclone_track_authentic(client):
    response = client.get("/api/cyclones/IMD-2020-AMPHAN/track")
    assert response.status_code == 200
    data = response.json()
    assert len(data["track"]) > 0
    assert data["track"][0]["latitude"] > 0


def test_get_cyclone_prediction_authentic(client):
    response = client.get("/api/cyclones/IMD-2020-AMPHAN/prediction")
    assert response.status_code == 200
    data = response.json()
    assert data["provenance"]["dataType"] == "AI_PREDICTION"


def test_get_cyclone_risk_authentic(client):
    response = client.get("/api/cyclones/IMD-2020-AMPHAN/risk")
    assert response.status_code == 200
    data = response.json()
    assert data["riskLevel"] in ("MINIMAL", "LOW", "MODERATE", "HIGH", "EXTREME")


def test_get_cyclone_digital_twin_authentic(client):
    response = client.get("/api/cyclones/IMD-2020-AMPHAN/digital-twin")
    assert response.status_code == 200
    data = response.json()
    assert data["radiusMaxWindKm"] > 0
    assert "R34" in data["quadrantRadii"]
    assert len(data["radialProfile"]) > 0


def test_get_ml_status(client):
    response = client.get("/api/ml/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("AWAITING_TRAINED_MODELS", "OPERATIONAL_RESEARCH")
    models = data["models"]
    assert "satellite_cyclone_detection" in models or "satellite_eye_detection" in models

