# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess


def get_git_commit_id() -> str:
    try:
        commit_id = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.PIPE).decode('utf-8').strip()
        return f"g{commit_id}"
    except subprocess.CalledProcessError as e:
        print(
            "Warning: Could not get git commit hash. Is this a git repository?"
        )
        print(f"Error: {e.stderr.decode('utf-8').strip()}")
        return "unknown"


def main() -> None:
    commit_id = get_git_commit_id()
    commit_filename = "commit.txt"

    with open(commit_filename, 'w') as f:
        f.write(commit_id)

    try:
        subprocess.run(['git', 'add', commit_filename],
                       check=True,
                       stderr=subprocess.PIPE)
        print(f"Staged '{commit_filename}' for commit.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to stage {commit_filename}.")
        print(f"Git error: {e.stderr.decode('utf-8').strip()}")


if __name__ == "__main__":
    if not os.path.exists('.git'):
        raise ValueError(
            "This script must be run from the root directory of git repository"
        )
    else:
        main()
