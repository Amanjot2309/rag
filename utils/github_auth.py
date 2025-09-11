import time
import os
import jwt
import requests

def generate_jwt():
    private_key_path = os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")
    app_id = os.getenv("GITHUB_APP_ID")

    with open(private_key_path, 'r') as pem_file:
        private_key = pem_file.read()

    payload = {
        "iat": int(time.time()),
        "exp": int(time.time()) + 600,  # 10 minutes
        "iss": app_id
    }

    jwt_token = jwt.encode(payload, private_key, algorithm="RS256")
    return jwt_token

def get_installation_access_token(installation_id):
    jwt_token = generate_jwt()

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }

    response = requests.post(url, headers=headers)

    if response.status_code != 201:
        raise Exception(f"Failed to get access token: {response.status_code} - {response.text}")

    return response.json()["token"]
