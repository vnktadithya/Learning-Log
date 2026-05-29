import requests
import re
import os
import time
from .Groq import generate_synthetic_telemetry
from .check_error_logs import contains_stack_trace_or_log
from .add_to_jsonl import append_to_jsonl

GITHUB_TOKEN = os.getenv("GitHub_PAT")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN. Get one from GitHub Developer Settings.")

url = "https://api.github.com/search/issues"

params = {
    "q": 'repo:langchain-ai/langchain is:pr is:merged label:"bug"',
    "per_page": 50,  # Fetch 50 items per page
    "page": 1,  # Initial page index
}

headers = {
    "User-Agent": "LangChainPRFetcher/1.0",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}


def fetch_real_rca_data(target_count=20):
    collected_examples = 0
    page = 1
    
    # We only want merged PRs with the 'bug' label
    while collected_examples < target_count:
        print(f"Fetching PR page {page}...")
        try:
            pr_response = requests.get(url=url, headers=headers, params=params, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Network error fetching page {page}: {e}. Retrying in 10s...")
            time.sleep(10)
            continue
        
        if pr_response.status_code != 200:
            print(f"GitHub API Error: {pr_response.text}")
            if pr_response.status_code == 403: # Rate limit hit
                print("Rate limit hit. Sleeping for 60 seconds...")
                time.sleep(60)
                continue
            break
            
        response_data = pr_response.json()
        prs = response_data.get("items", [])

        if not prs:
            print("No more items found in search results.")
            break 
            
        for pr in prs:

            if "pull_request" not in pr:
                print('No pull request in PR body.')
                continue
                
            body = pr.get("body") or ""
            
            # Look for the exact issue link syntax (e.g., Fixes #12345, Resolves #12345)
            matches = set(re.finditer(r"(?:[Ff]ixes|[Rr]esolves|[Cc]loses|[Ii]ssue)\s+#(\d+)", body))
            if not matches:
                print('No matches')
                continue

            for match in matches:
                issue_number = match.group(1)
                
                # Fetch the original issue
                issue_url = f"https://api.github.com/repos/langchain-ai/langchain/issues/{issue_number}"
                try:
                    issue_response = requests.get(issue_url, headers=headers, timeout=10)
                except requests.exceptions.RequestException:
                    print('hello')
                    continue
                
                if issue_response.status_code != 200:
                    print('hi')
                    continue
                    
                issue_data = issue_response.json()
                issue_text = issue_data.get("body") or ""
                body += f'#{issue_number}:\n{issue_text}\n'

                time.sleep(2)
                
            # FILTER: Does the issue actually contain a stack trace or raw log?
            if not contains_stack_trace_or_log(body):
                print("The related issue in the PR doesn't contain any stack trace or error log.")
                continue
                
            # call LLM agent to format the data and generate the response for the dataset.
            log_telemetry = generate_synthetic_telemetry(body)

            # defensive check against none type output
            if not log_telemetry:
                print(f"LLM failed to generate schema for PR: {pr.get('number', 'Unknown')}. Skipping.")
                continue
            
            #append data to log_telemetry file
            try:
                append_to_jsonl(
                    'Log_telemetry_dataset.jsonl', 
                    log_telemetry['raw_log'], 
                    log_telemetry['rca_schema']
                )
                collected_examples += 1
                print(f"Collected {collected_examples}/{target_count} examples.")
            except KeyError as e:
                print(f"Malformed LLM output missing key {e}. Skipping.")
                continue

            if collected_examples >= target_count:
                print("Dataset collected.")
                return
                            
        page += 1
        params["page"] = page
        time.sleep(2)

if __name__ == "__main__":
    fetch_real_rca_data(20)
