import sys
import time
import jwt
import requests

def generate_jwt(pem_path, app_id):
    with open(pem_path, 'rb') as pem_file:
        signing_key = pem_file.read()

    payload = {
        'iat': int(time.time()),
        'exp': int(time.time()) + 600,  # JWT expires after 10 minutes
        'iss': app_id  # GitHub App ID
    }

    jwt_token = jwt.encode(payload, signing_key, algorithm='RS256')
    return jwt_token

def get_installation_token(jwt_token, installation_id):
    headers = {
        "Authorization": f"Bearer {jwt_token}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/app/installations/{installation_id}/access_tokens"
    response = requests.post(url, headers=headers)

    if response.status_code != 201:
        print(f"Error: {response.status_code} - {response.text}")
        sys.exit(1)

    token_data = response.json()
    return token_data["token"]

def list_repo_issues(installation_token, owner, repo):
    headers = {
        "Authorization": f"token {installation_token}",
        "Accept": "application/vnd.github+json"
    }

    url = f"https://api.github.com/repos/{owner}/{repo}/issues"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"GitHub API error: {response.status_code} - {response.text}")
        sys.exit(1)

    return response.json()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python github_installation_auth.py <path_to_pem> <app_id> <installation_id>")
        sys.exit(1)

    pem_path = sys.argv[1]
    app_id = sys.argv[2]
    installation_id = sys.argv[3]

    owner = input("Enter the GitHub repo owner (org/user): ")
    repo = input("Enter the repository name: ")

    print("Generating JWT...")
    jwt_token = generate_jwt(pem_path, app_id)

    print("Requesting installation token...")
    installation_token = get_installation_token(jwt_token, installation_id)

    print("Fetching issues from repo...")
    issues = list_repo_issues(installation_token, owner, repo)

    for issue in issues:
        print(f"- #{issue['number']} {issue['title']}")
