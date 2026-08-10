from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root_hello() -> None:
    response = client.get("/")

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "hello from supportlens"
    assert "version" in body
