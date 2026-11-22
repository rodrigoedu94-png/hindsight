#!/usr/bin/env python3
"""Label PRs by the areas their diff touches.

Usage: python3 scripts/pr_labels.py <pr-number>

Labels (kept in line with the triage labels used for issues):
- "core": the diff touches the core service packages or shared tooling.
- "docs": the diff is documentation or docs tooling only.

Issues keep being labeled by scripts/triage-labels.py, which picks
core/integration via Gemini.

Needs the gh CLI (authenticated) and the repo in the GH_REPO environment
variable.
"""

import json
import os
import subprocess
import sys

CORE_DIRS = (
    "hindsight-api-slim/",
    "hindsight-api/",
    "hindsight-all/",
    "hindsight-all-slim/",
    "hindsight-all-npm/",
    "hindsight-cli/",
    "hindsight-embed/",
    "hindsight-extensions/",
    "hindsight-dev/",
    "hindsight-system-tests/",
    "hindsight-integration-tests/",
    "hindsight-system-evals/",
    "hindsight-integrations/",
    "hindsight-control-plane/",
    "hindsight-clients/",
    "scripts/",
)
DOC_DIRS = ("hindsight-docs/", "cookbook/", "skills/")


def gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def labels_for(names: list[str]) -> list[str]:
    labels = []
    if any(name.startswith(CORE_DIRS) for name in names):
        labels.append("core")
    if any(name.startswith(DOC_DIRS) for name in names):
        labels.append("docs")
    return labels


def main() -> None:
    owner, repo = os.environ["GH_REPO"].split("/")
    number = int(sys.argv[1])
    files = json.loads(gh("api", f"repos/{owner}/{repo}/pulls/{number}/files?per_page=100"))
    labels = labels_for([f["filename"] for f in files])
    for label in labels:
        gh("api", "-X", "POST", f"repos/{owner}/{repo}/issues/{number}/labels", "-f", f"labels[]={label}")
    print(f"pr #{number}: labels {labels if labels else '(none)'}")


if __name__ == "__main__":
    main()
