import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app


@pytest.fixture
def test_client():
    settings.TESTING = True
    with TestClient(app) as client:
        yield client
    settings.TESTING = False


def test_health_endpoints(test_client: TestClient):
    # Root health probe for cloud load balancers
    resp = test_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cyclone-x"

    # API health probe
    api_resp = test_client.get("/api/health")
    assert api_resp.status_code == 200
    assert api_resp.json()["status"] == "healthy"


def test_security_headers_and_correlation_id(test_client: TestClient):
    resp = test_client.get("/health")
    assert resp.status_code == 200
    headers = resp.headers

    assert "x-request-id" in headers
    assert headers.get("x-content-type-options") == "nosniff"
    assert headers.get("x-frame-options") == "DENY"
    assert headers.get("x-xss-protection") == "1; mode=block"
    assert headers.get("referrer-policy") == "strict-origin-when-cross-origin"


def test_auth_login_success(test_client: TestClient):
    payload = {
        "username": settings.ADMIN_USERNAME,
        "password": settings.ADMIN_PASSWORD,
    }
    resp = test_client.post("/api/auth/login", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == settings.ADMIN_USERNAME
    assert data["user"]["role"] == "admin"


def test_auth_login_invalid_credentials(test_client: TestClient):
    payload = {
        "username": settings.ADMIN_USERNAME,
        "password": "wrong_password_attempt",
    }
    resp = test_client.post("/api/auth/login", json=payload)
    assert resp.status_code == 401
    assert "Incorrect username or password" in resp.json()["message"]


def test_user_registration_and_privilege_escalation_defense(test_client: TestClient):
    # Public registration must not grant admin or operator roles even if requested
    payload = {
        "username": "researcher_test",
        "email": "researcher@example.com",
        "password": "SecurePassword123!",
        "role": "admin",  # Attempt privilege escalation
        "full_name": "Test Researcher",
    }
    resp = test_client.post("/api/auth/register", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["username"] == "researcher_test"
    # Role must be downgraded to viewer
    assert data["role"] == "viewer"


def test_rbac_admin_route_protection(test_client: TestClient):
    # 1. Unauthenticated access must be rejected
    unauth_resp = test_client.get("/api/auth/users")
    assert unauth_resp.status_code == 401

    # 2. Login as regular viewer user
    login_resp = test_client.post(
        "/api/auth/login",
        json={"username": "researcher_test", "password": "SecurePassword123!"},
    )
    viewer_token = login_resp.json()["access_token"]
    viewer_headers = {"Authorization": f"Bearer {viewer_token}"}

    # 3. Viewer accessing admin-only user catalog should receive 403 Forbidden
    forbidden_resp = test_client.get("/api/auth/users", headers=viewer_headers)
    assert forbidden_resp.status_code == 403

    # 4. Admin accessing user catalog should succeed with 200 OK
    admin_login = test_client.post(
        "/api/auth/login",
        json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD},
    )
    admin_token = admin_login.json()["access_token"]
    admin_headers = {"Authorization": f"Bearer {admin_token}"}

    admin_resp = test_client.get("/api/auth/users", headers=admin_headers)
    assert admin_resp.status_code == 200
    assert isinstance(admin_resp.json(), list)


def test_idor_protection_on_user_profiles(test_client: TestClient):
    # Login as viewer user
    login_resp = test_client.post(
        "/api/auth/login",
        json={"username": "researcher_test", "password": "SecurePassword123!"},
    )
    viewer_data = login_resp.json()
    viewer_id = viewer_data["user"]["id"]
    viewer_headers = {"Authorization": f"Bearer {viewer_data['access_token']}"}

    # Access own profile by ID should succeed
    own_resp = test_client.get(f"/api/auth/users/{viewer_id}", headers=viewer_headers)
    assert own_resp.status_code == 200
    assert own_resp.json()["id"] == viewer_id

    # Access admin profile (IDOR probe) should be rejected with 403 Forbidden
    other_resp = test_client.get("/api/auth/users/user-admin-default-01", headers=viewer_headers)
    assert other_resp.status_code == 403
    assert "Access denied" in other_resp.json()["message"]


def test_input_sanitization_on_cyclone_route(test_client: TestClient):
    # Malicious injection attempt in path parameter
    malicious_id = "amphan'; DROP TABLE cyclones;--"
    resp = test_client.get(f"/api/cyclones/{malicious_id}")
    assert resp.status_code == 400
    assert "disallowed characters" in resp.json()["message"]

    # Legitimate identifier should not be blocked by sanitizer
    valid_resp = test_client.get("/api/cyclones/BOB-01-2020")
    assert valid_resp.status_code in (200, 404)
