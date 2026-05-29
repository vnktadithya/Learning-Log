import requests
import re
import json
import os
import time

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

def contains_stack_trace_or_log(text): #specifically for python
    if not text:
        return False

    # 1. Check for standard Python Tracebacks
    if "Traceback (most recent call last):" in text:
        return True

    # 2. Check for common Python error endings (e.g., ValueError: ..., TypeError: ...)
    # This catches the last line of a stack trace even if the header was omitted
    error_pattern = (
        r"\b(?:[A-Z][a-zA-Z]+(?:Error|Exception|Fail)):\s+.+"
    )
    if re.search(error_pattern, text):
        return True

    # 3. Check for typical log patterns (e.g., 2026-05-28 12:00:00, or [WARNING], or INFO:)
    log_pattern = r"(?:\d{4}-\d{2}-\d{2}|\b(?:DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b[:\]\s])"
    if re.search(log_pattern, text, re.IGNORECASE):
        return True

    return False


def fetch_real_rca_data(target_count=20):
    collected_examples = 0
    page = 1
    
    # We only want merged PRs with the 'bug' label
    while collected_examples < target_count:
        print(f"Fetching PR page {page}...")
        pr_response = requests.get(url = url, headers = headers, params = params)
        
        if pr_response.status_code != 200:
            print(f"GitHub API Error: {pr_response.text}")
            break
            
        prs = pr_response.json()
        if not prs:
            break # No more PRs
            
        for pr in prs:
            if not pr.get("merged_at"):
                continue # Skip unmerged PRs
                
            body = pr.get("body") or ""
            
            # Look for the exact issue link syntax (e.g., Fixes #12345, Resolves #12345)
            matches = set(re.finditer(r"(?:[Ff]ixes|[Rr]esolves|[Cc]loses|[Ii]ssue)\s+#(\d+)", body))
            if not matches:
                continue

            for match in matches:
                
                issue_number = match.group(1)
                
                # Fetch the original issue
                issue_url = f"https://api.github.com/repos/langchain-ai/langchain/issues/{issue_number}"
                issue_response = requests.get(issue_url, headers=headers)
                
                if issue_response.status_code != 200:
                    continue
                    
                issue_data = issue_response.json()
                issue_body += f'#{issue_number}:\n{issue_data.get("body")}\n' or ""
                
            # FILTER: Does the issue actually contain a stack trace or raw log?
            if not contains_stack_trace_or_log(issue_body):
                continue
                
            # print(f"Valid Issue Found: #{issue_number} (From PR #{pr['number']})")
            
            # ==========================================
            # YOUR TURN TO ENGINEER:
            # 1. Take 'issue_body' (The User Prompt / Raw Log)
            # 2. Take 'body' (The PR Description / The Resolution)
            # 3. Call your LLM API (Groq/Gemini) to generate the RCA JSON schema based on the PR.
            # 4. Format them into the {"messages": [...]} schema.
            # 5. Append to esql_rca_dataset_final.jsonl
            # ==========================================
            
            # TODO: Implement LLM API Call here
            # TODO: Implement JSONL writing here
            
            collected_examples += 1
            if collected_examples >= target_count:
                print("Target reached.")
                break
                
            time.sleep(1) # Respect API limits
            
        page += 1

if __name__ == "__main__":
    fetch_real_rca_data(20)


#fetch the PR's which are closed, having the label as bug through pagination using GitHub API
# filter them: filter the PR's which actually contains the link to the exact issue they are solving
# use the link to navigate to the exact issue and fetch the info from the page
# filter these issues: hold the ones that exactly contains stack trace or the raw logs
# feed a PR with its related issue info to LLM and let it generate the exact data required for fine tuning
# store the data to .jsonl file