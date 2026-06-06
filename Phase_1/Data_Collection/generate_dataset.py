import requests
import re
import os
import time
from .LLM_Generator import generate_synthetic_telemetry
from .check_error_logs import contains_stack_trace_or_log
from .add_to_jsonl import append_to_jsonl

GITHUB_TOKEN = os.getenv("GitHub_PAT")
if not GITHUB_TOKEN:
    raise ValueError("Missing GITHUB_TOKEN. Get one from GitHub Developer Settings.")

url = "https://api.github.com/search/issues"

headers = {
    "User-Agent": "InfrastructureTelemetryFetcher/1.0",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json",
}

TARGET_REPOS = [        # Repositories written in python
   # "langchain-ai/langchain",
   # "celery/celery",          
   # "apache/airflow",         
   # "dask/dask",              
   # "fastapi/fastapi",       
    "pydantic/pydantic",
    "django/django",
    "PrefectHQ/prefect",
    "dagster-io/dagster",
    "encode/uvicorn",
    "pytest-dev/pytest",
    "ansible/ansible",
    "encode/httpx"       
]

def fetch_real_rca_data(target_count):
    collected_examples = 0
    page = 1
    
    # We only want merged PRs with the 'bug' label
    for repo in TARGET_REPOS:
        if collected_examples >= target_count:
            break

        page = 1

        params = {
                "q": f'repo:{repo} is:pr is:merged',
                "per_page": 50,  
                "page": page,  
            }

        print(f'Initiating scan on {repo}:')

        while collected_examples < target_count:
            print(f"Fetching PR page {page} in repo {repo}")


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
                print(f"   -> Exhausted search results for {repo}. Moving to next repository.")
                break
                
            for pr in prs:

                if "pull_request" not in pr:
                    print('No pull request in PR body.')
                    continue

                body = 'PULL REQUEST AND RELATED ISSUES:\n'  
                body += pr.get("body") or ""
                
                # Look for the exact issue link syntax (e.g., Fixes #12345, Resolves #12345)
                matches = set(re.finditer(r"(?:[Ff]ixes|[Rr]esolves|[Cc]loses|[Ii]ssue)\s+#(\d+)", body))
                if not matches:
                    print('No matches')
                    continue

                for match in matches:
                    issue_number = match.group(1)
                    
                    # Fetch the original issue
                    issue_url = f"https://api.github.com/repos/{repo}/issues/{issue_number}"

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
                        
                    # call LLM to format the data and generate the response for the dataset.
                    log_telemetry = generate_synthetic_telemetry(body)

                    # defensive check against none type output
                    if not log_telemetry:
                        print(f"LLM failed to generate schema for PR: {pr.get('number', 'Unknown')}. Skipping.")
                        continue

                    rca = log_telemetry.get('rca_schema', {}).get('root_cause_analysis', '')
                    if rca.lower().replace(" ", "_") == 'insufficient_data':
                        print(f"LLM rejected PR {pr.get('number', 'Unknown')} due to insufficient context. Skipping.")
                        continue
                    
                    #append data to log_telemetry file
                    try:
                        append_to_jsonl(
                            'Phase_1/Data_Collection/Log_telemetry_dataset.jsonl', 
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

            if page >= 3:
                break                  
            page += 1
            params["page"] = page
            time.sleep(2)


fetch_real_rca_data(20)
