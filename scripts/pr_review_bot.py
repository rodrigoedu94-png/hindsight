#!/usr/bin/env python3
"""Post a first-look review comment on each opened/reopened PR.

Usage: python3 scripts/pr_review_bot.py <pr-number>

Needs the gh CLI (authenticated) and the repo in the GH_REPO environment
variable.
"""

import json
import os
import subprocess
import sys


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def main() -> None:
    owner, repo = os.environ["GH_REPO"].split("/")
    number = int(sys.argv[1])
    pr = json.loads(gh("api", f"repos/{owner}/{repo}/pulls/{number}"))
    files = json.loads(gh("api", f"repos/{owner}/{repo}/pulls/{number}/files?per_page=100"))
    touched = sorted({f["filename"].split("/")[0] for f in files})
    additions = sum(f["additions"] for f in files)
    deletions = sum(f["deletions"] for f in files)
    body = (
        f"Auto-review: {pr['title']}\n"
        f"Changed {len(files)} file(s): +{additions} / -{deletions}\n"
        f"Touched: {', '.join(touched) if touched else '(none listed)'}"
    )
    gh("pr", "comment", str(number), "--body", body)
    print(f"pr #{number}: review comment posted")


if __name__ == "__main__":
    main()
