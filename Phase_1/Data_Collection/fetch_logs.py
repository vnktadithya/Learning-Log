import requests
import os

GitHub_PAT = os.getenv('GitHub_PAT')

url = "https://api.github.com/search/issues"

params = {
    "q": 'repo:kubernetes/kubernetes is:pr is:merged label:"kind/bug"'
}

headers = {
    "User-Agent": "KubernetesPRFetcher/1.0",
    "Authorization": f"Bearer {GitHub_PAT}",
    "Accept": "application/vnd.github.v3+json"
}

response = requests.get(url, params = params, headers = headers)
if response.status_code == 200:
    data = response.json()
    items = data.get('items', [])
    
    if not items:
        print("Still no items found. Check your query.")
    else:
        
        for i, item in enumerate(items[20:30]):
            
            issue_number = item.get('number')
            print(f"=== ISSUE #{issue_number} ===")
            print(f"TITLE: {item.get('title')}\n")
            print(f"BODY:\n{item.get('body')}...\n") # Truncated for readability
            
            # Check if there is a linked Pull Request
            if 'pull_request' in item:
                print(f"RESOLUTION PR LINK: {item['pull_request'].get('html_url')}\n")
            else:
                print("RESOLUTION: No direct PR linked in the payload.")

            print('-'*100 + '\n')
else:
    print(f"Error {response.status_code}: {response.text}")