import re
from typing import Optional, Tuple

def parse_action(string: str) -> Optional[Tuple[str, str]]:
    pattern = r'Action:\s*([A-Za-z0-9_]+)\s*\[(.*?)\]'

    match = re.search(pattern, string)

    if not match:
        print('Parsing error. LLM output did not match expected output fromat.')
        return None

    tool = match.group(1).strip()
    arg = match.group(2).strip()

    return tool, arg

print(parse_action('Action: Search[Colorado orogeny]'))