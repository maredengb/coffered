#!/usr/bin/env python3
"""Check explicit GitHub CLA confirmations. Never sign, merge, or execute PR code."""
from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

CONTEXT = "cofferdb/cla"
SHA = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LOGIN = re.compile(r"[A-Za-z0-9-]+(?:\[bot\])?\Z")
MAX_PRS = 200
MAX_COMMITS = 250  # The REST PR-commits endpoint has this documented ceiling.
MAX_COMMENTS = 5000
MAX_AUTHORS = 100
MAX_REQUESTS = 2500
RUN_SECONDS = 480
ROOT = Path(__file__).resolve().parents[2]


class Hold(Exception):
    """A fixed, public diagnostic code. Never include API bodies or credentials."""


def require(condition, code):
    if not condition:
        raise Hold(code)


def positive(value):
    return type(value) is int and value > 0


def timestamp(value):
    require(isinstance(value, str), "invalid_timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        require(result.utcoffset() == dt.timedelta(0), "invalid_timestamp")
        return result
    except (ValueError, TypeError):
        raise Hold("invalid_timestamp") from None


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(text):
    # No case folding, quote stripping, Markdown extraction, or fuzzy assent.
    require(isinstance(text, str), "invalid_comment_body")
    return text.replace("\r\n", "\n").strip()


def load_config(path, root=ROOT):
    try:
        config = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        raise Hold("invalid_configuration") from None
    require(isinstance(config, dict) and type(config.get("enabled")) is bool,
            "invalid_configuration")
    require(config.get("schema_version") == 1, "invalid_configuration")
    require(config.get("repository") == "maredengb/coffered" and
            config.get("repository_id") == 1422246026 and
            config.get("recipient_id") == 167590852, "wrong_repository_identity")
    for key, filename in (("agreement", "CLA.md"), ("notice", "CLA-PRIVACY.md")):
        item = config.get(key)
        require(isinstance(item, dict) and isinstance(item.get("version"), str) and
                re.fullmatch(r"[0-9]+\.[0-9]+", item["version"]) and
                isinstance(item.get("sha256"), str) and SHA256.fullmatch(item["sha256"]),
                "invalid_document_identity")
        try:
            data = (root / filename).read_bytes()
        except OSError:
            raise Hold("missing_document") from None
        require(digest(data) == item["sha256"], "document_hash_mismatch")
        if config["enabled"]:
            inactive_notices = (
                b"PREPARED_NOT_ACTIVE", b"DRAFT FOR REVIEW",
                b"This CLA is not yet in effect.",
                b"Online CLA confirmations are not enabled.",
            )
            require(not any(notice in data for notice in inactive_notices),
                    "document_not_active")
            expected = rf"https://github\.com/maredengb/coffered/blob/[0-9a-f]{{40}}/{re.escape(filename)}"
            require(isinstance(item.get("url"), str) and re.fullmatch(expected, item["url"]),
                    "invalid_document_url")
    if config["enabled"]:
        require(positive(config.get("issue_number")) and positive(config.get("issue_id")),
                "invalid_confirmation_issue")
        require(config.get("default_branch") == "main", "unsupported_default_branch")
        timestamp(config.get("recipient_reference_time"))
        timestamp(config.get("effective_at"))
        require(timestamp(config["recipient_reference_time"]) <= timestamp(config["effective_at"]),
                "invalid_activation_order")
        require(config.get("privacy_ready") is True and config.get("owner_records_ready") is True,
                "activation_prerequisites_missing")
    return config


def confirmation(config, number, head):
    repo = config["repository"]
    a, n = config["agreement"], config["notice"]
    return (f'I have read and agree to the CofferDB Contributor License Agreement version {a["version"]}, SHA256 {a["sha256"]}, at {a["url"]}, for repository {config["repository_id"]} ({repo}).\n'
            'I keep my copyright and grant the stated commercial, closed-source and other-license rights. I have reviewed the patent, future-contribution, succession and governing-law terms. I confirm that I have authority to license my contribution, including any necessary rights-holder permission; this confirmation does not license another person\'s work.\n'
            f'My initial covered contribution is my material in pull request https://github.com/{repo}/pull/{number} at commit {head}. The agreement also covers my later intentional contributions to this Project as stated in that version.\n'
            f'I have read the confirmation-data notice version {n["version"]}, SHA256 {n["sha256"]}, at {n["url"]}.\n'
            'Separately, I choose to publish this confirmation and its GitHub account attribution in this public thread, and I understand that others may read and copy it.')


def owner_confirmation(config, number, head):
    # A scoped assertion of existing rights, not a contract with oneself or a role exemption.
    return (f'I have reviewed my material in https://github.com/{config["repository"]}/pull/{number} at commit {head}. '
            'I confirm that I control the rights needed for the Project\'s current license and the licensing scope described in the CLA, and have retained the supporting rights review privately. '
            'This statement covers only my material at this commit, not another contributor\'s work. '
            f'I have read the confirmation-data notice SHA256 {config["notice"]["sha256"]}. '
            'Separately, I choose to publish this rights-review statement and its GitHub account attribution in this public thread.')


def config_digest(config):
    return digest(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())


def activation_notice(config, number, merge_sha):
    a, n = config["agreement"], config["notice"]
    return (f'CofferDB Contributor License Agreement invitation. Configuration SHA256 {config_digest(config)}.\n'
            f'Source: https://github.com/{config["repository"]}/pull/{number} @ {merge_sha}.\n'
            f'Repository: {config["repository"]} (ID {config["repository_id"]}). Recipient account ID: {config["recipient_id"]}; fixed reference time: {config["recipient_reference_time"]}.\n'
            f'Agreement version {a["version"]}, SHA256 {a["sha256"]}: {a["url"]}\n'
            f'Data notice version {n["version"]}, SHA256 {n["sha256"]}: {n["url"]}\n'
            'Please read both documents and ask any questions before agreeing. You retain copyright and grant the stated commercial, closed-source and other-license rights. Review the patent, future-contribution, lawful-succession and governing-law terms. Existing Apache-2.0 permissions remain in force.\n'
            'Confirmations and GitHub attribution are public. Do not post identity documents or private contact details. No separate identity form or external CLA service is required.\n'
            'For your first contribution, generate the exact text with python3 -B .github/cla/check.py --confirmation PR_NUMBER FULL_HEAD_SHA and personally post it in this issue. Replace both command arguments with your actual PR number and full head commit. Later contributions reuse a valid confirmation as described in the agreement.\n'
            'Automation checks records; it never signs for you.')


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise Hold("api_redirect_refused")


class Api:
    """Fixed GitHub origin; no arbitrary endpoint or response-URL following."""
    def __init__(self, token):
        require(bool(token), "missing_token")
        self.token = token
        self.started = time.monotonic()
        self.requests = 0
        self.opener = urllib.request.build_opener(NoRedirect())

    def call(self, method, path, body=None):
        require(method in ("GET", "POST"), "unsupported_method")
        require(path.startswith("/") and not path.startswith("//") and
                not any(c in path for c in "\r\n#"), "invalid_api_path")
        require(path == "/graphql" or path.startswith("/repos/maredengb/coffered/") or
                path == "/repos/maredengb/coffered" or
                re.fullmatch(r"/users/[A-Za-z0-9-]+(?:%5Bbot%5D)?", path), "invalid_api_path")
        require(method == "GET" or path == "/graphql" or
                re.fullmatch(r"/repos/maredengb/coffered/statuses/[0-9a-f]{40}", path),
                "write_endpoint_refused")
        self.requests += 1
        require(self.requests <= MAX_REQUESTS and time.monotonic()-self.started < RUN_SECONDS,
                "api_budget_exhausted")
        headers = {"Authorization": "Bearer " + self.token,
                   "Accept": "application/vnd.github+json",
                   "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "cofferdb-cla-check"}
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            req = urllib.request.Request("https://api.github.com"+path, data=data,
                                         headers=headers, method=method)
            with self.opener.open(req, timeout=15) as response:
                raw = response.read(8*1024*1024+1)
            require(len(raw) <= 8*1024*1024, "api_response_too_large")
            value = json.loads(raw)
            if path == "/graphql":
                require(isinstance(value, dict) and not value.get("errors") and
                        isinstance(value.get("data"), dict), "graphql_incomplete")
                return value["data"]
            return value
        except (urllib.error.URLError, OSError, ValueError):
            raise Hold("api_request_failed") from None

    def get(self, path):
        return self.call("GET", path)

    def post(self, path, body):
        return self.call("POST", path, body)


AUTHORS_QUERY = """query($owner:String!,$name:String!,$oid:String!,$after:String) {
  repository(owner:$owner,name:$name) { databaseId
    object(expression:$oid) { ... on Commit { oid message author { email }
      authors(first:100,after:$after) { totalCount pageInfo { hasNextPage endCursor }
        nodes { email user { databaseId login } } }
    } }
  }
}"""


class Checker:
    def __init__(self, config, api, run_url, dry_run=False):
        self.c, self.api, self.run_url, self.dry_run = config, api, run_url, dry_run
        self.prefix = "/repos/"+config["repository"]
        self.authors_cache = {}
        self.account_cache = {}
        self.pr_cache = {}
        self.confirmation_cache = {}
        self.records = []
        self.results = []
        self.write_failures = False
        self.activation_record = None
        self.workflow_sha = os.environ.get("GITHUB_WORKFLOW_SHA") if os.environ.get("GITHUB_ACTIONS") == "true" else None

    def verify_revision(self):
        if self.workflow_sha is not None:
            require(SHA.fullmatch(self.workflow_sha), "invalid_workflow_revision")
            current = self.api.get(f"{self.prefix}/commits/main")
            require(current.get("sha") == self.workflow_sha, "trusted_revision_changed")

    def verify_documents(self):
        for key, filename in (("agreement", "CLA.md"), ("notice", "CLA-PRIVACY.md")):
            doc = self.c[key]
            revision = doc["url"].split("/")[-2]
            value = self.api.get(f"{self.prefix}/contents/{filename}?ref={revision}")
            require(isinstance(value, dict) and value.get("type") == "file" and
                    value.get("path") == filename and value.get("encoding") == "base64" and
                    isinstance(value.get("content"), str), "invalid_published_document")
            try:
                raw = base64.b64decode(value["content"].replace("\n", ""), validate=True)
            except (ValueError, binascii.Error):
                raise Hold("invalid_published_document") from None
            require(digest(raw) == doc["sha256"], "published_document_hash_mismatch")

    def account(self, user):
        require(isinstance(user, dict) and positive(user.get("id")) and
                isinstance(user.get("login"), str) and LOGIN.fullmatch(user["login"]) and
                user.get("type") == "User", "unknown_or_nonhuman_account")
        identity = user["id"]
        if identity not in self.account_cache:
            actual = self.api.get("/users/"+urllib.parse.quote(user["login"], safe=""))
            require(isinstance(actual, dict) and actual.get("id") == identity and
                    actual.get("type") == "User", "account_identity_changed")
            self.account_cache[identity] = True
        return identity

    def pages(self, endpoint, limit):
        items, ids = [], set()
        for page in range(1, limit//100+2):
            sep = "&" if "?" in endpoint else "?"
            rows = self.api.get(f"{self.prefix}{endpoint}{sep}per_page=100&page={page}")
            require(isinstance(rows, list) and len(rows) <= 100, "invalid_pagination")
            for row in rows:
                require(isinstance(row, dict), "invalid_api_record")
                identity = row.get("id", row.get("sha"))
                require(identity is not None and identity not in ids, "unstable_pagination")
                ids.add(identity)
                items.append(row)
                require(len(items) <= limit, "pagination_limit_exceeded")
            if len(rows) < 100:
                return items
        raise Hold("pagination_limit_exceeded")

    def verify_repository_identity(self):
        repo = self.api.get(self.prefix)
        require(repo.get("id") == self.c["repository_id"] and
                repo.get("full_name") == self.c["repository"] and
                repo.get("default_branch") == "main" and
                repo.get("owner", {}).get("id") == self.c["recipient_id"],
                "repository_identity_changed")

    def verify_repository(self):
        self.verify_revision()
        self.verify_documents()
        self.verify_repository_identity()
        issue = self.api.get(f'{self.prefix}/issues/{self.c["issue_number"]}')
        require(issue.get("id") == self.c["issue_id"] and "pull_request" not in issue and
                issue.get("html_url") == f'https://github.com/{self.c["repository"]}/issues/{self.c["issue_number"]}' and
                issue.get("state") == "open" and not issue.get("locked"), "confirmation_issue_unavailable")
        require(timestamp(self.c["effective_at"]) <= dt.datetime.now(dt.timezone.utc),
                "activation_not_effective")

    def verify_activation(self, comments):
        self.activation_record = None
        for comment in comments:
            if comment.get("user", {}).get("id") != self.c["recipient_id"]:
                continue
            try:
                self.valid_record(comment, after_activation=False)
                body = normalized(comment["body"])
                match = re.search(r'Source: https://github.com/maredengb/coffered/pull/([1-9][0-9]{0,18}) @ ([0-9a-f]{40})\.', body)
                if not match or body != activation_notice(self.c, int(match[1]), match[2]):
                    continue
                number, merge_sha = int(match[1]), match[2]
                p = self.api.get(f"{self.prefix}/pulls/{number}")
                require(p.get("number") == number and p.get("merged") is True and
                        p.get("merge_commit_sha") == merge_sha and
                        p.get("merged_by", {}).get("id") == self.c["recipient_id"] and
                        p.get("base", {}).get("repo", {}).get("id") == self.c["repository_id"] and
                        p.get("base", {}).get("ref") == "main", "invalid_activation_merge")
                require(timestamp(p.get("merged_at")) < timestamp(comment["created_at"]),
                        "activation_predates_merge")
                ancestry = self.api.get(f"{self.prefix}/compare/{merge_sha}...main")
                require(ancestry.get("status") in {"ahead", "identical"} and
                        ancestry.get("merge_base_commit", {}).get("sha") == merge_sha,
                        "activation_not_on_main")
                value = self.api.get(f"{self.prefix}/contents/.github/cla/config.json?ref={merge_sha}")
                require(value.get("type") == "file" and value.get("path") == ".github/cla/config.json" and
                        value.get("encoding") == "base64" and isinstance(value.get("content"), str),
                        "invalid_activation_configuration")
                try:
                    frozen = json.loads(base64.b64decode(value["content"].replace("\n", ""), validate=True))
                except (KeyError, TypeError, ValueError):
                    raise Hold("invalid_activation_configuration") from None
                require(config_digest(frozen) == config_digest(self.c), "activation_configuration_changed")
                self.activation_record = self.record(comment, "activation")
                self.activation_record.update(pr=number, merge_sha=merge_sha)
                return
            except Hold as error:
                if str(error).startswith("api_"):
                    raise
        raise Hold("owner_activation_announcement_required")

    def pr(self, number, fresh=False):
        require(positive(number), "invalid_pr_number")
        if fresh or number not in self.pr_cache:
            p = self.api.get(f"{self.prefix}/pulls/{number}")
            require(isinstance(p, dict) and p.get("number") == number and
                    p.get("base", {}).get("repo", {}).get("id") == self.c["repository_id"] and
                    p.get("base", {}).get("ref") == "main" and
                    isinstance(p.get("head", {}).get("sha"), str) and SHA.fullmatch(p["head"]["sha"]) and
                    positive(p.get("commits")) and p["commits"] <= MAX_COMMITS,
                    "pr_scope_or_commit_limit")
            self.pr_cache[number] = p
        return self.pr_cache[number]

    def authors(self, sha):
        if sha in self.authors_cache:
            return self.authors_cache[sha]
        require(isinstance(sha, str) and SHA.fullmatch(sha), "invalid_commit")
        after, seen, actors, emails, total = None, set(), set(), set(), None
        message = primary = None
        for _ in range(2):
            data = self.api.post("/graphql", {"query": AUTHORS_QUERY, "variables": {
                "owner": "maredengb", "name": "coffered", "oid": sha, "after": after}})
            repo = data.get("repository")
            require(isinstance(repo, dict) and repo.get("databaseId") == self.c["repository_id"],
                    "graphql_repository_mismatch")
            commit = repo.get("object")
            require(isinstance(commit, dict) and commit.get("oid") == sha and
                    isinstance(commit.get("message"), str), "missing_commit")
            connection = commit.get("authors")
            require(isinstance(connection, dict) and positive(connection.get("totalCount")) and
                    connection["totalCount"] <= MAX_AUTHORS, "author_count_limit")
            if total is None:
                total, message, primary = connection["totalCount"], commit["message"], commit.get("author", {}).get("email")
            require(total == connection["totalCount"] and message == commit["message"], "unstable_author_page")
            nodes = connection.get("nodes")
            require(isinstance(nodes, list) and nodes, "missing_authors")
            for node in nodes:
                require(isinstance(node, dict) and isinstance(node.get("email"), str) and node["email"],
                        "unknown_author")
                user = node.get("user")
                require(isinstance(user, dict), "unlinked_author")
                ident = self.account({"id": user.get("databaseId"), "login": user.get("login"), "type": "User"})
                key = (node["email"].lower(), ident)
                require(key not in seen, "duplicate_author_page")
                seen.add(key); actors.add(ident); emails.add(node["email"].lower())
            info = connection.get("pageInfo")
            require(isinstance(info, dict) and type(info.get("hasNextPage")) is bool, "invalid_author_page")
            if not info["hasNextPage"]:
                require(len(seen) == total, "incomplete_authors")
                break
            cursor = info.get("endCursor")
            require(isinstance(cursor, str) and cursor and cursor != after, "invalid_author_cursor")
            after = cursor
        else:
            raise Hold("author_pagination_limit")
        require(isinstance(primary, str) and primary.lower() in emails, "missing_primary_author")
        # Do not guess accounts from emails. Compare trailers with the API's complete author set.
        for line in message.splitlines():
            if "co-authored-by" in line.lower():
                match = re.fullmatch(r"\s*Co-authored-by:\s+[^<>\r\n]+\s+<([^<>\s]+@[^<>\s]+)>\s*", line, re.I)
                require(match is not None and match[1].lower() in emails, "unresolved_coauthor_trailer")
        self.authors_cache[sha] = actors
        return actors

    def pr_authors(self, number, required_commit=None):
        p = self.pr(number)
        commits = self.pages(f"/pulls/{number}/commits", MAX_COMMITS)
        require(len(commits) == p["commits"] and p["head"]["sha"] in {r.get("sha") for r in commits},
                "incomplete_pr_commits")
        if required_commit:
            require(required_commit in {r.get("sha") for r in commits}, "initial_commit_not_in_pr")
        actors = set()
        for commit in commits:
            sha = commit.get("sha")
            found = self.authors(sha)
            if required_commit and sha != required_commit:
                comparison = self.api.get(f"{self.prefix}/compare/{sha}...{required_commit}")
                require(isinstance(comparison, dict), "invalid_initial_ancestry")
                if comparison.get("status") not in {"ahead", "identical"}:
                    continue
                require(comparison.get("merge_base_commit", {}).get("sha") == sha,
                        "invalid_initial_ancestry")
            actors.update(found)
        require(actors, "missing_pr_authors")
        # Detect a moving PR rather than caching authors from a different head/base.
        latest = self.pr(number, fresh=True)
        require(latest["head"]["sha"] == p["head"]["sha"] and latest["commits"] == p["commits"] and
                latest["base"]["sha"] == p["base"]["sha"], "pr_changed_during_check")
        return actors

    def valid_record(self, comment, after_activation=True):
        require(isinstance(comment, dict) and positive(comment.get("id")) and
                isinstance(comment.get("body"), str) and len(comment["body"]) <= 10000,
                "invalid_confirmation_record")
        require(comment.get("created_at") == comment.get("updated_at"), "confirmation_edited")
        created = timestamp(comment.get("created_at"))
        require(timestamp(self.c["effective_at"]) <= created <= dt.datetime.now(dt.timezone.utc),
                "confirmation_outside_activation")
        if after_activation:
            require(self.activation_record is not None and
                    timestamp(self.activation_record["created_at"]) < created,
                    "confirmation_before_opening")
        require(comment.get("html_url") == f'https://github.com/{self.c["repository"]}/issues/{self.c["issue_number"]}#issuecomment-{comment["id"]}',
                "confirmation_wrong_thread")
        require(comment.get("issue_url") == f'https://api.github.com{self.prefix}/issues/{self.c["issue_number"]}',
                "confirmation_wrong_thread")
        return self.account(comment.get("user"))

    def covers(self, record, current_pr):
        created = timestamp(current_pr.get("created_at"))
        confirmed = timestamp(record["created_at"])
        require(created <= dt.datetime.now(dt.timezone.utc), "invalid_pr_creation_time")
        if record["initial_pr"] == current_pr["number"]:
            return created <= confirmed and record["initial_head"] == current_pr["head"]["sha"]
        # Same-second ordering is ambiguous. Older unrelated PRs need their own scope.
        return created > confirmed

    def signed(self, actor, comments, current_pr):
        for comment in comments:
            if comment.get("user", {}).get("id") != actor:
                continue
            key = comment.get("id")
            if key in self.confirmation_cache:
                cached = self.confirmation_cache[key]
                if cached and self.covers(cached, current_pr):
                    return cached
                continue
            record = None
            try:
                self.valid_record(comment)
                body = normalized(comment["body"])
                m = re.search(r'My initial covered contribution is my material in pull request https://github\.com/maredengb/coffered/pull/([1-9][0-9]{0,18}) at commit ([0-9a-f]{40})\.', body)
                if m and body == confirmation(self.c, int(m[1]), m[2]):
                    initial = self.pr(int(m[1]))
                    require(timestamp(initial.get("created_at")) <= timestamp(comment["created_at"]),
                            "confirmation_predates_pr")
                    require(actor in self.pr_authors(int(m[1]), required_commit=m[2]), "signer_not_initial_author")
                    record = self.record(comment, "cla")
                    record.update(initial_pr=int(m[1]), initial_head=m[2])
            except Hold as error:
                # Only malformed/edited comments may be skipped; API incompleteness must not turn green.
                if str(error) in {"api_request_failed", "api_budget_exhausted", "graphql_incomplete",
                                  "api_response_too_large", "api_redirect_refused"}:
                    raise
            self.confirmation_cache[key] = record
            if record and self.covers(record, current_pr):
                return record
        raise Hold("missing_valid_confirmation")

    def record(self, comment, kind):
        return {"kind": kind, "id": comment["id"], "url": comment["html_url"],
                "account_id": comment["user"]["id"], "login": comment["user"]["login"],
                "created_at": comment["created_at"], "updated_at": comment["updated_at"],
                "body": comment["body"], "body_sha256": digest(comment["body"].encode())}

    def owner_record(self, number, head, comments):
        for comment in comments:
            if comment.get("user", {}).get("id") != self.c["recipient_id"]:
                continue
            try:
                self.valid_record(comment)
                if normalized(comment["body"]) == owner_confirmation(self.c, number, head):
                    require(timestamp(self.pr(number).get("created_at")) <= timestamp(comment["created_at"]),
                            "confirmation_predates_pr")
                    return self.record(comment, "owner_rights_review")
            except Hold as error:
                if str(error).startswith("api_"):
                    raise
        raise Hold("owner_rights_review_required")

    def fresh_records(self, records):
        require(self.activation_record is not None, "owner_activation_announcement_required")
        for record in [self.activation_record, *records]:
            comment = self.api.get(f'{self.prefix}/issues/comments/{record["id"]}')
            require(comment.get("id") == record["id"] and
                    self.valid_record(comment, after_activation=record["kind"] != "activation") == record["account_id"] and
                    comment["body"] == record["body"] and
                    comment["created_at"] == record["created_at"], "confirmation_changed_during_check")

    def status(self, head, state, code):
        require(SHA.fullmatch(head) and state in {"pending", "failure", "success"}, "invalid_status")
        if not self.dry_run:
            try:
                self.api.post(f"{self.prefix}/statuses/{head}", {
                    "state": state, "context": CONTEXT, "description": code[:140], "target_url": self.run_url})
            except Hold:
                self.write_failures = True
                raise

    def check_pr(self, number, comments, expected_head=None):
        p, records = self.pr(number, fresh=True), []
        head = p["head"]["sha"]
        require(expected_head is None or head == expected_head, "pr_changed_during_check")
        require(p.get("state") == "open", "pr_not_open")
        self.status(head, "pending", "Checking explicit confirmations; no ownership guarantee")
        try:
            actors = self.pr_authors(number)
            for actor in sorted(actors):
                records.append(self.owner_record(number, head, comments) if actor == self.c["recipient_id"]
                               else self.signed(actor, comments, p))
            self.fresh_records(records)
            latest = self.pr(number, fresh=True)
            require(latest["head"]["sha"] == head and latest["base"]["sha"] == p["base"]["sha"] and
                    latest.get("state") == "open", "pr_changed_during_check")
            self.verify_revision()
            self.status(head, "success", "Confirmations checked; owner must review rights and merge manually")
            self.records.extend(records)
            result = {"pr": number, "head": head, "result": "success", "record_ids": [r["id"] for r in records]}
        except Hold as error:
            self.status(head, "failure", str(error))
            result = {"pr": number, "head": head, "result": "failure", "reason": str(error)}
        self.results.append(result)
        return result

    def run(self):
        self.verify_repository_identity()
        pulls = self.pages("/pulls?state=open&base=main&sort=created&direction=asc", MAX_PRS)
        # Invalidate every discovered head before reading the mutable ledger.
        for p in pulls:
            head = p.get("head", {}).get("sha")
            require(isinstance(head, str) and SHA.fullmatch(head), "invalid_pr_head")
            self.status(head, "pending", "Rechecking the confirmation ledger")
        # Known heads are pending even if documents, issue availability, or policy fail.
        self.verify_repository()
        comments = self.pages(f'/issues/{self.c["issue_number"]}/comments', MAX_COMMENTS)
        self.verify_activation(comments)
        head_counts = {}
        for p in pulls:
            head = p["head"]["sha"]
            head_counts[head] = head_counts.get(head, 0) + 1
        for p in pulls:
            try:
                require(head_counts[p["head"]["sha"]] == 1, "duplicate_pr_head_requires_review")
                self.check_pr(p["number"], comments, expected_head=p["head"]["sha"])
            except Hold as error:
                head = p["head"]["sha"]
                try:
                    self.status(head, "failure", str(error))
                except Hold:
                    pass
                self.results.append({"pr": p["number"], "head": head, "result": "failure", "reason": str(error)})
        require(not self.write_failures, "status_write_failed")
        return self.results


def write_export(directory, checker, config_path):
    dest = Path(directory).expanduser().resolve()
    # Export belongs in owner-controlled storage, never the checkout or an Actions artifact.
    require(not os.environ.get("GITHUB_ACTIONS") and not dest.is_relative_to(ROOT) and
            not any((parent / ".git").exists() for parent in (dest, *dest.parents)),
            "unsafe_export_location")
    try:
        dest.mkdir(mode=0o700, parents=False, exist_ok=False)
        files = {"records.json": json.dumps({"exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "mode": "local_read_only_recheck", "results": checker.results,
                   "checker_sha256": digest(Path(__file__).read_bytes()),
                   "workflow_sha256": digest((ROOT/".github/workflows/cla.yml").read_bytes()),
                   "activation": checker.activation_record,
                   "records": list({r["id"]: r for r in checker.records}.values())}, indent=2).encode(),
                 "config.json": Path(config_path).read_bytes(),
                 "CLA.md": (ROOT/"CLA.md").read_bytes(), "CLA-PRIVACY.md": (ROOT/"CLA-PRIVACY.md").read_bytes()}
        files["SHA256SUMS"] = ''.join(f'{digest(data)}  {name}\n' for name, data in files.items()).encode()
        for name, data in files.items():
            fd = os.open(dest/name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
    except OSError:
        raise Hold("export_failed") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--dry-run", action="store_true", help="Read APIs but never write statuses")
    parser.add_argument("--export-dir", help="New private directory; only with --dry-run, outside Actions")
    texts = parser.add_mutually_exclusive_group()
    texts.add_argument("--confirmation", nargs=2, metavar=("PR", "HEAD"), help="Print text, never post it")
    texts.add_argument("--owner-confirmation", nargs=2, metavar=("PR", "HEAD"),
                       help="Print a scoped owner rights statement, never post it")
    texts.add_argument("--activation-notice", nargs=2, metavar=("PR", "MERGE_SHA"),
                       help="Print the confirmation invitation; never post it")
    args = parser.parse_args(argv)
    try:
        c = load_config(args.config)
        if not c["enabled"]:
            print("PREPARED_NOT_ACTIVE: no API calls, signatures, or status writes.")
            return 0
        require(not args.export_dir or args.dry_run, "export_requires_dry_run")
        if args.confirmation or args.owner_confirmation or args.activation_notice:
            number, head = args.confirmation or args.owner_confirmation or args.activation_notice
            require(number.isdigit() and int(number)>0 and SHA.fullmatch(head), "invalid_confirmation_target")
            render = activation_notice if args.activation_notice else owner_confirmation if args.owner_confirmation else confirmation
            print(render(c, int(number), head))
            return 0
        action = os.environ.get("GITHUB_ACTIONS") == "true"
        if action:
            event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
            require(event.get("repository", {}).get("id") == c["repository_id"], "wrong_event_repository")
            require(os.environ.get("GITHUB_EVENT_NAME") in {"pull_request_target", "issue_comment", "workflow_dispatch", "push"}, "unsupported_event")
            require(os.environ.get("GITHUB_REF") == "refs/heads/main" and
                    SHA.fullmatch(os.environ.get("GITHUB_WORKFLOW_SHA", "")), "untrusted_workflow_ref")
            run_id = os.environ.get("GITHUB_RUN_ID", "")
            require(run_id.isdigit(), "invalid_run_identity")
            run_url = f'https://github.com/{c["repository"]}/actions/runs/{run_id}'
        else:
            require(args.dry_run, "local_status_writes_refused")
            run_url = f'https://github.com/{c["repository"]}/issues/{c["issue_number"]}'
        checker = Checker(c, Api(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")), run_url, args.dry_run)
        results = checker.run()
        if args.export_dir:
            write_export(args.export_dir, checker, args.config)
        for result in results:
            print(f'PR {result["pr"]}: {result["result"]}' + (f' ({result["reason"]})' if "reason" in result else ""))
        return 1 if any(r["result"] != "success" for r in results) else 0
    except Hold as error:
        print("CLA check held: " + str(error), file=sys.stderr)
        return 1
    except (KeyError, TypeError, ValueError, OSError, AttributeError):
        print("CLA check held: invalid_or_incomplete_input", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
