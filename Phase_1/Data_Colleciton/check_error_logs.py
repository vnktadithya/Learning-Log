import re

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