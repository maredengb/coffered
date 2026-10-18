# GitHub-native CLA confirmation checker

The checker is **disabled** (`config.json`: `enabled: false`). No
signing thread, required check, or merge restriction is activated by these files.
The public license remains Apache-2.0. Read [CLA.md](../../CLA.md), the
[data notice](../../CLA-PRIVACY.md) and [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Contribution flow after activation

For a first contribution, read the exact agreement and data notice linked by the
activation announcement. Personally post the complete confirmation in the
designated public issue. The text identifies its versions and hashes, the
repository, and the initial PR and commit. No additional identity form or external
CLA service is used. You can generate the text without posting anything:

```sh
python3 -B .github/cla/check.py --confirmation PR_NUMBER FULL_HEAD_SHA
```

The disabled configuration does not generate text for signing. After
activation, the same account's unchanged confirmation is reused for subsequent
contributions under that version. A renamed account is matched by numeric ID.
Confirm the first PR at its final head. If that head changes, post a new
head-specific confirmation; it requires no extra identity information. Later-created
PRs reuse an existing confirmation automatically. A PR already open before that
confirmation, or created in the same timestamp second, needs its own explicit
scope. This conservatively avoids extending the grant to older unrelated work.
Changes to the agreement require new consent. Edited, deleted, mismatched or
unverifiable confirmations do not automatically clear later PRs. Posting the
comment is the contributor's action; automation never posts a signature.

All identified contributors must be covered. The checker uses GitHub's author
and coauthor associations; these are not proof of authorship or rights. Unknown
accounts, bots, unresolved coauthor trailers, employer-owned work and other
rights questions need review. A PR opener cannot sign for other people.

The Recipient does not contract with themselves. For their own material, the
Recipient personally posts a separate, **PR-and-head-specific** rights-review
statement after checking and privately retaining the rights basis. This is not a
blanket owner exemption, a CLA on someone else's behalf, or a bot exception:

```sh
python3 -B .github/cla/check.py --owner-confirmation PR_NUMBER FULL_HEAD_SHA
```

The final owner review checks actual sources, applicable third-party notices and
any additional rights-holder authority. Only the owner manually merges.

## What is checked

- Frozen repository and numeric Recipient IDs, activation time, agreement and
  notice hashes, immutable published document contents, and the designated issue.
- Exact affirmative text from the actual human GitHub comment author. Quotes,
  code blocks, partial text, placeholders, edited comments and bot comments do
  not count. Only line endings and outer whitespace are normalized.
- The initial PR includes the stated commit, and the signer's material belongs
  to that commit or its PR ancestors. Later additions cannot retroactively make
  someone an author of the initially confirmed submission. If rewriting the
  initial PR makes that scope unverifiable, a new explicit confirmation is needed.
- All current PR commits and author/coauthor pages. There are explicit bounds:
  200 open PRs, 250 commits per PR, 100 authors per commit, and 5,000 comments in
  the confirmation issue. Exceeding a bound requires review; it is not a pass.
  No account is guessed from an email address.
- Current PR head and base before and after reading records, and fresh copies of
  the confirmations immediately before success. In Actions, the workflow
  revision must still be the current default-branch revision.

`cofferdb/cla` is explicitly written to the PR **head SHA**. A successful workflow
job on the default branch is not this status. Statuses are commit-scoped: multiple
open PRs sharing a head are held for review. A person checking a merge must also
verify that the run's PR number matches the intended PR.

Every relevant run rechecks open PRs targeting `main`, first setting their known
heads to pending. API errors, incomplete pages, unknown authors or write failures
never produce a successful result for the affected check. A service outage can
prevent invalidating an earlier success; event delivery and cross-system reads
are not atomic. **Immediately before each merge, request a fresh manual run and
verify its completion, current policy revision, PR number, head and records.**
A stale green status is not permission to merge.

## Workflow and permissions

The workflow runs on trusted `main` revisions for PR-target events, confirmation
issue comments, main pushes and manual dispatch. All issue-comment runs rescan
the fixed confirmation ledger, even when the triggering comment is elsewhere;
this prevents an unrelated queued event from replacing a needed rescan with a
no-op. Comments outside the designated issue can never be confirmations. The checkout Action is pinned
to a full commit, checks out `github.workflow_sha`, and does not persist credentials.
It never checks out, installs dependencies from, or runs the proposed PR content.
No event text is inserted into a shell command. The Python checker uses the
standard library and makes requests only to fixed GitHub API paths.

Permissions are `contents: read`, `pull-requests: read`, `issues: read`, and
`statuses: write`; all others are disabled. It cannot write a branch, approve a
PR, merge, administer the repository or sign for a contributor. No owner PAT,
write deploy key, extra OAuth application or external signing database is used.

GitHub branch rules must separately require the actual `cofferdb/cla` status,
with its expected Actions source where supported. Source binding may distinguish
an app rather than an individual workflow; people able to modify trusted
workflows or write statuses remain part of the trust boundary. The repository
owner's control of credentials and settings cannot be replaced by a CLA checker.
Existing collaborators, apps, keys, bypasses and automatic-merge settings must be
inventoried before restrictions are represented as effective.

## Activation procedure

1. Review the final agreement, notice, checker and workflow. Resolve the
   Recipient's private identity record, necessary disclosures, confidential
   contact route and data-handling arrangements. Do not publish private identity
   records, credentials or exceptional authorization documents.
2. The owner manually merges the trusted implementation, still disabled. Do not
   configure a required check before a real producer is available.
3. Publish the activation documents at immutable commit URLs. Bind
   their exact SHA256 values, versions, fixed Recipient reference time, repository
   ID, issue ID/number and effective time in `config.json`. The activation notice
   must reproduce this identity and prominently explain the important terms.
4. Once the activation prerequisites are satisfied, the owner manually merges the
   enabled configuration. Conduct real-fork checks with a consenting human,
   who personally posts any actual confirmation. Do not substitute fake records
   or a bot's signature for this verification.
5. Confirm the status appears on the exact PR head, including failure cases, and
   then require that status. Separately verify owner-only manual merge controls,
   without silently removing existing access. Inspect the final settings again.

A configuration change is a reviewed change. A changed agreement or notice must
not inherit old confirmations silently. Keep earlier terms and evidence; do not
rewrite prior permissions or treat a repository transfer as a change of legal party.

## Local controls and private evidence

Run deterministic controls without contacting GitHub:

```sh
python3 -B -m unittest discover -s .github/cla -p 'test_*.py'
```

After activation, the owner can read and export evidence without writing statuses:

```sh
# Supply GH_TOKEN through a secure local environment, never a command argument.
python3 -B .github/cla/check.py --dry-run --export-dir /PRIVATE/NEW_DIRECTORY
```

Use a new directory outside every Git checkout and outside Actions. The exporter
creates a mode-0700 directory and mode-0600 files with minimal confirmation
records, exact document/config copies, current PR results and hashes. It refuses
existing destinations and does not upload artifacts. It omits commit-author
emails and unrelated account-profile fields. The owner handles access-controlled
backups, retention review and lawful rights requests. Platform timestamps and
file hashes are evidence aids, not independent identity or trusted-time certification.
The export identifies the local checker and workflow files and labels the results
as a local read-only recheck; it does not claim to be a hosted Actions execution.
Retain the actual completed Actions run link separately for the manual merge.
A failed or incomplete export is not a complete acceptance record.

These controls do not prove legal ownership, replace source review, or certify
statutory signature validity. Automated failures do not revoke existing licenses.
