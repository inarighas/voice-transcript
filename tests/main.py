from fastapi.testclient import TestClient
from app.main import app
# Transcription, FileAudio, FlushedAudio

client = TestClient(app)


def test_main_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {'message': "Hello World"}


# File transcription


def test_transcribe_inexistent_file():
    response = client.post("/transcribe_file",
                           json={"content_format": "wav",
                                 "name": "bonjourds"})
    assert response.status_code == 404
    assert response.json() == {"detail": "File not found"}


def test_transcribe_unsupported_format():
    response = client.post("/transcribe_file",
                            json={
                                "content_format": "mp3",
                                "name": "bonjour"
                            })
    assert response.status_code == 404
    assert response.json() == {"detail": "Unsupported format"}


def test_transcribe_corrupt_file():
    response = client.post("/transcribe_file",
                           json={
                               "content_format": "wav",
                               "name": "corrupt_file"
                           })
    assert response.status_code == 404
    assert response.json() == {"detail": "File error"}

# def test_transcribe_file_with_differrent_sr():
#     response = client.post("/transcribe_file",
#                            headers={"X-Token": "coneofsilence"})
#     assert response.status_code == 404
#     assert response.json() == {"detail": "Item not found"}


def test_transcribe_file():
    response = client.post("/transcribe_file",
                           json={
                               "content_format": "wav",
                               "name": "bonjour"
                           }
                           )
    assert response.status_code == 200
    assert response.json() == {"content": "bonjour tout le monde",
                               "word_rate": 58.8938989601546,
                               "sentiment": "",
                               "sentiment_probas":
                                   "[\'testing\', \'\', \'\', \'\']",
                               "status": "success"}
