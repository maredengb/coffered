# GitHub-native CLA confirmation checker

This checker verifies GitHub account confirmations of the
[Contributor License Agreement](../../CLA.md). The public distribution remains
Apache-2.0. Read the [data notice](../../CLA-PRIVACY.md) and
[contribution guide](../../CONTRIBUTING.md).

## Confirmation text

For your first covered contribution, personally post the complete confirmation
in the designated public thread. It identifies the agreement and notice versions
and hashes, the repository, and your initial PR and commit. No additional identity
form or external CLA service is used. Generate the text without posting anything:

```sh
python3 -B .github/cla/check.py --confirmation PR_NUMBER FULL_HEAD_SHA
```

Replace the arguments with your PR number and full commit hash. Each relevant
contributor confirms for their own material; a PR opener cannot sign for others.
Later contributions within the agreement's scope reuse the existing confirmation.
Changes to the agreement require new consent. Automation never signs for you.

The Recipient does not contract with themselves. For their own material, a
separate statement identifies the PR, commit and rights basis; it does not grant
another person's rights:

```sh
python3 -B .github/cla/check.py --owner-confirmation PR_NUMBER FULL_HEAD_SHA
```

## Technical scope

The checker binds repository and account IDs, exact agreement and notice contents,
and the designated confirmation records. It verifies affirmative text and the
covered contribution against GitHub's author and coauthor associations. It never
guesses an account from an email address. Account associations and check results
do not prove authorship, ownership or employer authorization.

The `cofferdb/cla` status belongs to the PR head commit. Unknown authors,
unverifiable records and incomplete API results require review. The scan limits
are 200 open PRs, 250 commits per PR, 100 authors per commit and 5,000 comments in
the confirmation thread; exceeding a limit does not produce a pass.

The workflow uses trusted repository code and never executes proposed PR content.
Permissions are `contents: read`, `pull-requests: read`, `issues: read` and
`statuses: write`; all others are disabled. The checker cannot write a branch,
approve a PR, administer the repository or sign for a contributor. It uses Python's
standard library and fixed GitHub API paths, with no additional signing service.

## Local checks

Run deterministic controls without contacting GitHub:

```sh
python3 -B -m unittest discover -s .github/cla -p 'test_*.py'
```

A read-only check makes no status writes:

```sh
python3 -B .github/cla/check.py --dry-run
```

Provide the read credential through `GH_TOKEN` in the environment, not a command
argument or a tracked file. Check failures do not revoke existing licenses.
