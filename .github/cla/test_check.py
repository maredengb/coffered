#!/usr/bin/env python3
"""Offline tests for explicit confirmation checks; all contributor identities are synthetic."""
from __future__ import annotations

import base64
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.parse

SPEC = importlib.util.spec_from_file_location("cla_check", Path(__file__).with_name("check.py"))
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)
REPO = "maredengb/coffered"
PREFIX = "/repos/" + REPO
REPOSITORY_ID = 1422246026
OWNER_ID = 167590852
OWNER = {"id": OWNER_ID, "login": "maredengb", "type": "User"}
ALICE = {"id": 900101, "login": "synthetic-alice", "type": "User"}
BOB = {"id": 900102, "login": "synthetic-bob", "type": "User"}
SHA1, SHA2, SHA3, BASE = (character * 40 for character in "123b")
BEFORE = "2023-12-31T00:00:00Z"
OPENED = "2023-12-30T00:00:00Z"
MERGED = "2023-12-29T00:00:00Z"
WHEN = "2024-01-01T00:00:00Z"
AFTER = "2024-01-02T00:00:00Z"
DOCUMENTS = {"CLA.md": b"Synthetic agreement bytes\n", "CLA-PRIVACY.md": b"Synthetic notice bytes\n"}


def config():
    return {
        "schema_version": 1, "enabled": True, "repository": REPO,
        "repository_id": REPOSITORY_ID, "recipient_id": OWNER_ID,
        "default_branch": "main", "issue_number": 17, "issue_id": 900017,
        "recipient_reference_time": "2020-01-01T00:00:00Z",
        "effective_at": "2020-01-02T00:00:00Z",
        "privacy_ready": True, "owner_records_ready": True,
        "agreement": {"version": "1.0", "sha256": check.digest(DOCUMENTS["CLA.md"]),
                      "url": f"https://github.com/{REPO}/blob/{BASE}/CLA.md"},
        "notice": {"version": "1.0", "sha256": check.digest(DOCUMENTS["CLA-PRIVACY.md"]),
                   "url": f"https://github.com/{REPO}/blob/{BASE}/CLA-PRIVACY.md"},
    }


def pull(number=1, head=SHA1, commits=1, created_at=BEFORE):
    return {"id": 910000 + number, "number": number, "state": "open", "commits": commits,
            "created_at": created_at,
            "head": {"sha": head}, "base": {"sha": BASE, "ref": "main", "repo": {"id": REPOSITORY_ID}}}


def comment(c, actor=ALICE, number=1, head=SHA1, identity=920001, owner=False):
    return {"id": identity, "user": copy.deepcopy(actor),
            "body": (check.owner_confirmation(c, number, head) if owner else
                     check.confirmation(c, number, head)),
            "created_at": WHEN, "updated_at": WHEN,
            "html_url": f"https://github.com/{REPO}/issues/17#issuecomment-{identity}",
            "issue_url": f"https://api.github.com{PREFIX}/issues/17"}


def author_data(sha, actors=(ALICE,), message="Add a synthetic test fixture"):
    nodes = [{"email": f"{actor['login']}@example.invalid",
              "user": {"databaseId": actor["id"], "login": actor["login"]}} for actor in actors]
    return {"repository": {"databaseId": REPOSITORY_ID, "object": {
        "oid": sha, "message": message, "author": {"email": nodes[0]["email"]},
        "authors": {"totalCount": len(nodes), "nodes": nodes,
                    "pageInfo": {"hasNextPage": False, "endCursor": None}}}}}


class FakeApi:
    """In-memory GitHub responses. Unexpected requests fail instead of reaching a network."""
    def __init__(self, c):
        self.c = c
        self.repo = {"id": REPOSITORY_ID, "full_name": REPO, "default_branch": "main",
                     "owner": {"id": OWNER_ID}}
        self.issue = {"id": 900017, "state": "open", "locked": False,
                      "html_url": f"https://github.com/{REPO}/issues/17"}
        self.users = {actor["login"]: copy.deepcopy(actor) for actor in (OWNER, ALICE, BOB)}
        self.pulls = {1: pull()}
        self.commit_lists = {1: [{"sha": SHA1}]}
        self.graphql = {SHA1: author_data(SHA1)}
        self.comments = [comment(c)]
        self.activation = comment(c, actor=OWNER, identity=920000)
        self.activation.update(body=check.activation_notice(c, 99, BASE),
                               created_at=OPENED, updated_at=OPENED)
        self.activation_pr = {"number": 99, "merged": True, "merged_at": MERGED,
                              "merge_commit_sha": BASE, "merged_by": OWNER,
                              "base": {"ref": "main", "repo": {"id": REPOSITORY_ID}}}
        self.activation_config = copy.deepcopy(c)
        self.activation_ancestry = {"status": "identical", "merge_base_commit": {"sha": BASE}}
        self.comparisons = {}
        self.main_sha = BASE
        self.documents = copy.deepcopy(DOCUMENTS)
        self.fresh_comments = {}
        self.statuses = []
        self.calls = []
        self.failures = {}
        self.responses = {}
        self.open_pulls = None

    def _override(self, method, path):
        self.calls.append((method, path))
        if (method, path) in self.failures:
            raise check.Hold(self.failures[(method, path)])
        if (method, path) in self.responses:
            values = self.responses[(method, path)]
            if values:
                return True, copy.deepcopy(values.pop(0))
        return False, None

    def get(self, path):
        found, value = self._override("GET", path)
        if found:
            return value
        parsed = urllib.parse.urlsplit(path)
        endpoint = parsed.path
        params = urllib.parse.parse_qs(parsed.query)
        if endpoint == PREFIX:
            return copy.deepcopy(self.repo)
        if endpoint == PREFIX + "/issues/17":
            return copy.deepcopy(self.issue)
        if endpoint == PREFIX + "/commits/main":
            return {"sha": self.main_sha}
        if endpoint == PREFIX + "/pulls/99":
            return copy.deepcopy(self.activation_pr)
        if endpoint == PREFIX + "/compare/" + BASE + "...main":
            return copy.deepcopy(self.activation_ancestry)
        if endpoint == PREFIX + "/contents/.github/cla/config.json":
            if params.get("ref") != [BASE]:
                raise AssertionError("Activation config must come from the exact merge commit")
            return {"type": "file", "path": ".github/cla/config.json", "encoding": "base64",
                    "content": base64.b64encode(json.dumps(self.activation_config).encode()).decode()}
        if endpoint.startswith(PREFIX + "/contents/"):
            filename = endpoint.removeprefix(PREFIX + "/contents/")
            if filename not in self.documents or params.get("ref") != [BASE]:
                raise AssertionError(f"Unexpected document: {path}")
            return {"type": "file", "path": filename, "encoding": "base64",
                    "content": base64.b64encode(self.documents[filename]).decode()}
        if endpoint.startswith(PREFIX + "/compare/"):
            relation = endpoint.removeprefix(PREFIX + "/compare/")
            if relation not in self.comparisons:
                raise AssertionError(f"Unexpected comparison: {path}")
            return copy.deepcopy(self.comparisons[relation])
        if endpoint.startswith("/users/"):
            login = urllib.parse.unquote(endpoint.removeprefix("/users/"))
            if login not in self.users:
                raise check.Hold("api_request_failed")
            return copy.deepcopy(self.users[login])
        if endpoint.startswith(PREFIX + "/issues/comments/"):
            identity = int(endpoint.rsplit("/", 1)[1])
            if identity in self.fresh_comments:
                value = self.fresh_comments[identity]
                if value is None:
                    raise check.Hold("api_request_failed")
                return copy.deepcopy(value)
            for item in self.comments + ([self.activation] if self.activation else []):
                if item["id"] == identity:
                    return copy.deepcopy(item)
            raise check.Hold("api_request_failed")
        if endpoint == PREFIX + "/pulls":
            rows = list(self.pulls.values()) if self.open_pulls is None else self.open_pulls
        elif endpoint == PREFIX + "/issues/17/comments":
            rows = self.comments + ([self.activation] if self.activation else [])
        elif endpoint.startswith(PREFIX + "/pulls/"):
            parts = endpoint.removeprefix(PREFIX + "/pulls/").split("/")
            number = int(parts[0])
            if len(parts) == 1:
                if number not in self.pulls:
                    raise check.Hold("api_request_failed")
                return copy.deepcopy(self.pulls[number])
            if parts[1:] != ["commits"]:
                raise AssertionError(f"Unexpected endpoint: {path}")
            rows = self.commit_lists[number]
        else:
            raise AssertionError(f"Unexpected endpoint: {path}")
        page = int(params.get("page", ["1"])[0])
        size = int(params.get("per_page", ["100"])[0])
        return copy.deepcopy(rows[(page - 1) * size:page * size])

    def post(self, path, body):
        found, value = self._override("POST", path)
        if found:
            return value
        if path == "/graphql":
            sha = body["variables"]["oid"]
            return copy.deepcopy(self.graphql[sha])
        if path.startswith(PREFIX + "/statuses/"):
            self.statuses.append((path.rsplit("/", 1)[1], copy.deepcopy(body)))
            return {"id": 930000 + len(self.statuses)}
        raise AssertionError(f"Unexpected write: {path}")


class OfflineTestCase(unittest.TestCase):
    def setUp(self):
        environment = mock.patch.dict(os.environ, {}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)


class CheckerTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.c = config()
        self.api = FakeApi(self.c)
        self.checker = check.Checker(self.c, self.api,
                                     f"https://github.com/{REPO}/actions/runs/123")
        self.checker.verify_activation([self.api.activation])

    def result(self, number=1):
        return self.checker.check_pr(number, self.api.comments)

    def assert_failed(self, reason=None):
        result = self.result()
        self.assertEqual("failure", result["result"])
        if reason:
            self.assertEqual(reason, result["reason"])
        self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])
        return result

    def test_changed_recipient_control_never_writes_a_status(self):
        self.api.repo["owner"]["id"] = BOB["id"]
        with self.assertRaisesRegex(check.Hold, "repository_identity_changed"):
            self.checker.run()
        self.assertEqual([], self.api.statuses)

    def test_first_explicit_confirmation_succeeds_on_exact_head(self):
        result = self.result()
        self.assertEqual("success", result["result"])
        self.assertEqual([920001], result["record_ids"])
        self.assertEqual([(SHA1, "pending"), (SHA1, "success")],
                         [(sha, body["state"]) for sha, body in self.api.statuses])
        self.assertTrue(all(body["context"] == "cofferdb/cla" for _, body in self.api.statuses))

    def test_existing_confirmation_covers_later_intentional_contribution(self):
        self.api.pulls[2] = pull(2, SHA2, created_at=AFTER)
        self.api.commit_lists[2] = [{"sha": SHA2}]
        self.api.graphql[SHA2] = author_data(SHA2)
        result = self.result(2)
        self.assertEqual("success", result["result"])
        self.assertEqual([920001], result["record_ids"])
        self.assertEqual(SHA2, self.api.statuses[-1][0])

    def test_old_or_same_second_unrelated_pr_is_not_covered(self):
        self.api.commit_lists[2] = [{"sha": SHA2}]
        self.api.graphql[SHA2] = author_data(SHA2)
        for created in (BEFORE, WHEN):
            with self.subTest(created=created):
                self.api.pulls[2] = pull(2, SHA2, created_at=created)
                result = self.result(2)
                self.assertEqual("failure", result["result"])
                self.assertEqual("missing_valid_confirmation", result["reason"])
        self.api.pulls[2] = pull(2, SHA2, created_at=AFTER)
        self.assertEqual("success", self.result(2)["result"])

    def test_cached_valid_record_does_not_expand_cross_pr_scope(self):
        self.assertEqual("success", self.result()["result"])
        for number, sha, created in ((2, SHA2, AFTER), (3, SHA3, BEFORE)):
            self.api.pulls[number] = pull(number, sha, created_at=created)
            self.api.commit_lists[number] = [{"sha": sha}]
            self.api.graphql[sha] = author_data(sha)
        self.assertEqual("success", self.result(2)["result"])
        result = self.result(3)
        self.assertEqual("failure", result["result"])
        self.assertEqual("missing_valid_confirmation", result["reason"])
        self.assertEqual("failure", self.api.statuses[-1][1]["state"])

    def test_cached_record_does_not_bypass_missing_current_pr_creation_time(self):
        self.assertEqual("success", self.result()["result"])
        del self.api.pulls[1]["created_at"]
        result = self.result()
        self.assertEqual("failure", result["result"])
        self.assertNotEqual("success", self.api.statuses[-1][1]["state"])

    def test_cached_record_rejects_future_current_pr_creation_time(self):
        self.assertEqual("success", self.result()["result"])
        for number, sha in ((1, SHA1), (2, SHA2)):
            with self.subTest(number=number):
                self.api.pulls[number] = pull(number, sha, created_at="2999-01-01T00:00:00Z")
                self.api.commit_lists[number] = [{"sha": sha}]
                self.api.graphql[sha] = author_data(sha)
                result = self.result(number)
                self.assertEqual("failure", result["result"])
                self.assertEqual("invalid_pr_creation_time", result["reason"])

    def test_initial_pr_new_head_requires_a_new_explicit_confirmation(self):
        self.assertEqual("success", self.result()["result"])
        self.api.pulls[1] = pull(1, SHA2, 2)
        self.api.commit_lists[1] = [{"sha": SHA1}, {"sha": SHA2}]
        self.api.graphql[SHA2] = author_data(SHA2)
        self.api.comparisons[SHA2 + "..." + SHA1] = {"status": "behind", "merge_base_commit": {"sha": SHA1}}
        self.api.comparisons[SHA1 + "..." + SHA2] = {"status": "ahead", "merge_base_commit": {"sha": SHA1}}
        result = self.result()
        self.assertEqual("failure", result["result"])
        self.assertEqual("missing_valid_confirmation", result["reason"])
        self.api.comments.append(comment(self.c, ALICE, 1, SHA2, identity=920002))
        result = self.result()
        self.assertEqual("success", result["result"])
        self.assertEqual([920002], result["record_ids"])

    def test_explicit_initial_scope_allows_same_second_pr_creation(self):
        self.api.pulls[1]["created_at"] = WHEN
        self.assertEqual("success", self.result()["result"])

    def test_initial_confirmation_cannot_predate_initial_pr(self):
        self.api.pulls[1]["created_at"] = AFTER
        self.assert_failed("missing_valid_confirmation")

    def test_initial_pr_creation_time_is_required_and_valid(self):
        for value in (None, "not-a-date", "2024-01-01T00:00:00", 42):
            with self.subTest(value=value):
                self.checker.confirmation_cache.clear()
                self.api.pulls[1]["created_at"] = value
                self.assert_failed("missing_valid_confirmation")
        del self.api.pulls[1]["created_at"]
        self.checker.confirmation_cache.clear()
        self.assert_failed("missing_valid_confirmation")

    def test_later_pr_creation_time_is_required_even_for_cached_record(self):
        self.assertEqual("success", self.result()["result"])
        self.api.commit_lists[2] = [{"sha": SHA2}]
        self.api.graphql[SHA2] = author_data(SHA2)
        for value in (None, "not-a-date", "2024-01-01T00:00:00", 42):
            with self.subTest(value=value):
                self.api.pulls[2] = pull(2, SHA2, created_at=value)
                result = self.result(2)
                self.assertEqual("failure", result["result"])
                self.assertEqual("invalid_timestamp", result["reason"])
        del self.api.pulls[2]["created_at"]
        self.assertEqual("failure", self.result(2)["result"])

    def test_owner_confirmation_cannot_predate_pr_or_omit_creation_time(self):
        self.api.graphql[SHA1] = author_data(SHA1, (OWNER,))
        self.api.comments = [comment(self.c, OWNER, owner=True)]
        for value in (AFTER, None, "not-a-date"):
            with self.subTest(value=value):
                self.api.pulls[1]["created_at"] = value
                self.assert_failed("owner_rights_review_required")
        del self.api.pulls[1]["created_at"]
        self.assert_failed("owner_rights_review_required")

    def test_multiple_authors_require_separate_confirmations(self):
        self.api.graphql[SHA1] = author_data(SHA1, (ALICE, BOB))
        self.assert_failed("missing_valid_confirmation")
        self.api.comments.append(comment(self.c, BOB, identity=920002))
        self.assertEqual("success", self.result()["result"])

    def test_owner_needs_current_pr_rights_review(self):
        self.api.graphql[SHA1] = author_data(SHA1, (OWNER,))
        self.assert_failed("owner_rights_review_required")
        self.api.comments.append(comment(self.c, OWNER, owner=True, identity=920002))
        self.assertEqual("success", self.result()["result"])

    def test_owner_statement_does_not_cover_other_authors(self):
        self.api.graphql[SHA1] = author_data(SHA1, (OWNER, BOB))
        self.api.comments = [comment(self.c, OWNER, owner=True)]
        self.assert_failed("missing_valid_confirmation")

    def test_owner_statement_is_bound_to_pr_head(self):
        self.api.graphql[SHA1] = author_data(SHA1, (OWNER,))
        for number, head in ((2, SHA1), (1, SHA2)):
            with self.subTest(number=number, head=head):
                self.api.comments = [comment(self.c, OWNER, number, head, owner=True)]
                self.assert_failed("owner_rights_review_required")

    def test_unrelated_or_altered_assent_never_passes(self):
        original = self.api.comments[0]["body"]
        changes = ["I agree", "> " + original, "````\n" + original + "\n````",
                   original + "\nI reject the commercial license.",
                   original.replace("I keep my copyright", "I transfer my copyright"),
                   original.replace("version 1.0", "version 0.9", 1),
                   original.replace(self.c["agreement"]["sha256"], "f" * 64),
                   original.replace(self.c["notice"]["sha256"], "f" * 64)]
        for body in changes:
            with self.subTest(body=body[:50]):
                self.checker.confirmation_cache.clear()
                self.api.comments[0]["body"] = body
                self.assert_failed("missing_valid_confirmation")

    def test_line_endings_and_outer_whitespace_do_not_change_assent(self):
        self.api.comments[0]["body"] = "\n" + self.api.comments[0]["body"].replace("\n", "\r\n") + "\n"
        self.assertEqual("success", self.result()["result"])

    def test_bot_comment_cannot_sign(self):
        self.api.comments[0]["user"]["type"] = "Bot"
        self.assert_failed("missing_valid_confirmation")

    def test_current_bot_account_is_not_accepted_as_human_author(self):
        self.api.users[ALICE["login"]]["type"] = "Bot"
        self.assert_failed("account_identity_changed")

    def test_account_id_prevents_username_reassignment(self):
        self.api.users[ALICE["login"]]["id"] = 999999
        self.assert_failed("account_identity_changed")

    def test_edited_or_removed_confirmation_fails(self):
        self.api.comments[0]["updated_at"] = "2024-01-02T00:00:00Z"
        self.assert_failed("missing_valid_confirmation")
        self.api.comments = []
        self.assert_failed("missing_valid_confirmation")

    def test_comment_from_other_thread_is_not_accepted(self):
        self.api.comments[0]["issue_url"] = f"https://api.github.com{PREFIX}/issues/18"
        self.assert_failed("missing_valid_confirmation")

    def test_confirmation_must_be_within_activation_dates(self):
        for value in ("2019-01-01T00:00:00Z", "2999-01-01T00:00:00Z", "not-a-date"):
            with self.subTest(value=value):
                self.checker.confirmation_cache.clear()
                self.api.comments[0]["created_at"] = self.api.comments[0]["updated_at"] = value
                self.assert_failed("missing_valid_confirmation")

    def test_signer_must_be_an_author_of_initial_contribution(self):
        self.api.pulls[1]["created_at"] = AFTER
        self.api.pulls[2] = pull(2, SHA2)
        self.api.commit_lists[2] = [{"sha": SHA2}]
        self.api.graphql[SHA2] = author_data(SHA2, (BOB,))
        self.api.comments[0] = comment(self.c, ALICE, 2, SHA2)
        self.assert_failed("missing_valid_confirmation")

    def test_initial_commit_must_belong_to_the_named_pr(self):
        self.api.comments[0] = comment(self.c, ALICE, 1, SHA2)
        self.assert_failed("missing_valid_confirmation")

    def test_later_author_cannot_retroactively_validate_initial_confirmation(self):
        self.api.pulls[1]["created_at"] = AFTER
        self.api.pulls[2] = pull(2, SHA3, 2)
        self.api.commit_lists[2] = [{"sha": SHA2}, {"sha": SHA3}]
        self.api.graphql[SHA2] = author_data(SHA2, (BOB,))
        self.api.graphql[SHA3] = author_data(SHA3, (ALICE,))
        self.api.comparisons[SHA3 + "..." + SHA2] = {"status": "behind", "merge_base_commit": {"sha": SHA2}}
        self.api.comments[0] = comment(self.c, ALICE, 2, SHA2)
        self.assert_failed("missing_valid_confirmation")

    def test_initial_scope_includes_proved_ancestor_contributions(self):
        self.api.pulls[1]["created_at"] = AFTER
        self.api.pulls[2] = pull(2, SHA3, 2)
        self.api.commit_lists[2] = [{"sha": SHA2}, {"sha": SHA3}]
        self.api.graphql[SHA2] = author_data(SHA2, (ALICE,))
        self.api.graphql[SHA3] = author_data(SHA3, (BOB,))
        self.api.comparisons[SHA2 + "..." + SHA3] = {"status": "ahead", "merge_base_commit": {"sha": SHA2}}
        self.api.comments[0] = comment(self.c, ALICE, 2, SHA3)
        self.assertEqual("success", self.result()["result"])

    def test_false_ancestor_claim_cannot_validate_initial_scope(self):
        self.api.pulls[1]["created_at"] = AFTER
        self.api.pulls[2] = pull(2, SHA3, 2)
        self.api.commit_lists[2] = [{"sha": SHA2}, {"sha": SHA3}]
        self.api.graphql[SHA2] = author_data(SHA2, (ALICE,))
        self.api.graphql[SHA3] = author_data(SHA3, (BOB,))
        self.api.comparisons[SHA2 + "..." + SHA3] = {"status": "ahead", "merge_base_commit": {"sha": BASE}}
        self.api.comments[0] = comment(self.c, ALICE, 2, SHA3)
        self.assert_failed("missing_valid_confirmation")

    def test_unknown_author_cannot_be_replaced_by_pr_submitter(self):
        self.api.graphql[SHA1]["repository"]["object"]["authors"]["nodes"][0]["user"] = None
        self.assert_failed("unlinked_author")

    def test_unresolved_coauthor_trailer_fails(self):
        self.api.graphql[SHA1] = author_data(SHA1, message="Change\n\nCo-authored-by: Nobody <nobody@example.invalid>")
        self.assert_failed("unresolved_coauthor_trailer")

    def test_malformed_coauthor_trailer_fails(self):
        self.api.graphql[SHA1] = author_data(SHA1, message="Co-authored-by: unknown")
        self.assert_failed("unresolved_coauthor_trailer")

    def test_known_coauthor_trailer_is_checked_against_all_accounts(self):
        self.api.graphql[SHA1] = author_data(SHA1, (ALICE, BOB),
                                           "Change\n\nCo-authored-by: Synthetic Bob <synthetic-bob@example.invalid>")
        self.api.comments.append(comment(self.c, BOB, identity=920002))
        self.assertEqual("success", self.result()["result"])

    def test_author_count_and_complete_connection_are_required(self):
        connection = self.api.graphql[SHA1]["repository"]["object"]["authors"]
        connection["totalCount"] = 2
        self.assert_failed("incomplete_authors")

    def test_duplicate_author_page_fails(self):
        connection = self.api.graphql[SHA1]["repository"]["object"]["authors"]
        connection["totalCount"] = 2
        connection["nodes"] *= 2
        self.assert_failed("duplicate_author_page")

    def test_missing_primary_author_fails(self):
        self.api.graphql[SHA1]["repository"]["object"]["author"]["email"] = "unmapped@example.invalid"
        self.assert_failed("missing_primary_author")

    def test_author_pagination_without_cursor_fails(self):
        connection = self.api.graphql[SHA1]["repository"]["object"]["authors"]
        connection["pageInfo"] = {"hasNextPage": True, "endCursor": None}
        self.assert_failed("invalid_author_cursor")

    def test_incomplete_commit_listing_fails(self):
        self.api.pulls[1]["commits"] = 2
        self.assert_failed("incomplete_pr_commits")

    def test_rest_pagination_handles_exact_page_and_limit(self):
        self.api.activation = None  # This test isolates a 100-comment pagination boundary.
        rows = [{"id": n + 1} for n in range(100)]
        self.api.comments = rows
        self.assertEqual(rows, self.checker.pages("/issues/17/comments", 100))
        self.assertIn(("GET", PREFIX + "/issues/17/comments?per_page=100&page=2"), self.api.calls)
        self.api.comments.append({"id": 101})
        with self.assertRaisesRegex(check.Hold, "pagination_limit_exceeded"):
            self.checker.pages("/issues/17/comments", 100)

    def test_duplicate_rest_page_fails(self):
        self.api.comments = [{"id": n + 1} for n in range(100)] + [{"id": 1}]
        with self.assertRaisesRegex(check.Hold, "unstable_pagination"):
            self.checker.pages("/issues/17/comments", 200)

    def test_more_than_one_hundred_records_in_a_page_fails(self):
        self.api.responses[("GET", PREFIX + "/issues/17/comments?per_page=100&page=1")] = [
            [{"id": n + 1} for n in range(101)]]
        with self.assertRaisesRegex(check.Hold, "invalid_pagination"):
            self.checker.pages("/issues/17/comments", 200)

    def test_moving_pr_head_and_base_never_get_success(self):
        original = pull()
        for field, new_value in (("head", SHA2), ("base", SHA3)):
            with self.subTest(field=field):
                changed = copy.deepcopy(original)
                changed[field]["sha"] = new_value
                self.api.responses[("GET", PREFIX + "/pulls/1")] = [original, changed]
                self.checker.pr_cache.clear()
                self.assert_failed("pr_changed_during_check")

    def test_deleted_confirmation_at_final_read_fails(self):
        self.api.fresh_comments[920001] = None
        self.assert_failed("api_request_failed")

    def test_changed_confirmation_at_final_read_fails(self):
        changed = copy.deepcopy(self.api.comments[0])
        changed["body"] += " altered"
        self.api.fresh_comments[920001] = changed
        self.assert_failed("confirmation_changed_during_check")

    def test_api_failure_does_not_fall_through_to_success(self):
        self.api.failures[("POST", "/graphql")] = "api_request_failed"
        self.assert_failed("api_request_failed")

    def test_status_permission_failure_is_fatal(self):
        self.api.failures[("POST", PREFIX + "/statuses/" + SHA1)] = "api_request_failed"
        with self.assertRaisesRegex(check.Hold, "api_request_failed"):
            self.checker.run()
        self.assertTrue(self.checker.write_failures)
        self.assertEqual([], self.api.statuses)

    def test_dry_run_reads_but_never_writes_status(self):
        self.checker.dry_run = True
        self.assertEqual("success", self.checker.run()[0]["result"])
        self.assertEqual([], self.api.statuses)
        self.assertTrue(any(path == "/graphql" for method, path in self.api.calls if method == "POST"))

    def test_repository_numeric_identity_is_checked(self):
        self.api.repo["id"] += 1
        with self.assertRaisesRegex(check.Hold, "repository_identity_changed"):
            self.checker.run()
        self.assertEqual([], self.api.statuses)

    def test_closed_or_locked_confirmation_issue_fails(self):
        for key, value in (("state", "closed"), ("locked", True)):
            with self.subTest(key=key):
                previous = self.api.issue[key]
                self.api.issue[key] = value
                with self.assertRaisesRegex(check.Hold, "confirmation_issue_unavailable"):
                    self.checker.run()
                self.api.issue[key] = previous

    def test_shared_head_across_open_prs_cannot_receive_success(self):
        self.api.pulls[2] = pull(2, SHA1)
        self.api.commit_lists[2] = [{"sha": SHA1}]
        try:
            self.checker.run()
        except check.Hold:
            pass
        self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])

    def test_preflight_api_failure_invalidates_known_old_success(self):
        self.assertEqual("success", self.result()["result"])
        previous_writes = len(self.api.statuses)
        self.api.failures[("GET", PREFIX + "/contents/CLA.md?ref=" + BASE)] = "api_request_failed"
        with self.assertRaisesRegex(check.Hold, "api_request_failed"):
            self.checker.run()
        new_states = [body["state"] for _, body in self.api.statuses[previous_writes:]]
        self.assertEqual(["pending"], new_states)
        self.assertEqual("pending", self.api.statuses[-1][1]["state"])

    def test_changed_published_document_blocks_the_run(self):
        self.api.documents["CLA.md"] = b"Different published agreement"
        with self.assertRaisesRegex(check.Hold, "published_document_hash_mismatch"):
            self.checker.run()
        self.assertEqual("pending", self.api.statuses[-1][1]["state"])

    def test_invalid_published_document_encoding_fails(self):
        endpoint = PREFIX + "/contents/CLA.md?ref=" + BASE
        self.api.responses[("GET", endpoint)] = [{
            "type": "file", "path": "CLA.md", "encoding": "base64", "content": "not valid base64!"}]
        with self.assertRaisesRegex(check.Hold, "invalid_published_document"):
            self.checker.run()
        self.assertEqual("pending", self.api.statuses[-1][1]["state"])

    def test_old_workflow_revision_cannot_validate_new_policy(self):
        self.checker.workflow_sha = SHA2
        with self.assertRaisesRegex(check.Hold, "trusted_revision_changed"):
            self.checker.run()
        self.assertEqual("pending", self.api.statuses[-1][1]["state"])

    def test_main_change_before_success_fails(self):
        self.checker.workflow_sha = BASE
        self.api.responses[("GET", PREFIX + "/commits/main")] = [{"sha": BASE}, {"sha": SHA2}]
        result = self.checker.run()[0]
        self.assertEqual("failure", result["result"])
        self.assertEqual("trusted_revision_changed", result["reason"])
        self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])

    def test_head_changes_after_open_pr_list_remains_blocked(self):
        self.api.open_pulls = [pull(1, SHA2)]
        result = self.checker.run()[0]
        self.assertEqual("failure", result["result"])
        self.assertEqual("pr_changed_during_check", result["reason"])
        self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])

    def test_private_export_has_restricted_permissions_and_no_author_emails(self):
        self.assertEqual("success", self.result()["result"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "documents"
            root.mkdir()
            for filename, data in DOCUMENTS.items():
                (root / filename).write_bytes(data)
            workflow = root / ".github/workflows/cla.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("name: Synthetic workflow fixture\n")
            configuration = root / "config.json"
            configuration.write_text(json.dumps(self.c))
            destination = Path(directory) / "private-export"
            with mock.patch.object(check, "ROOT", root):
                check.write_export(destination, self.checker, configuration)
            self.assertEqual(0o700, destination.stat().st_mode & 0o777)
            for path in destination.iterdir():
                self.assertEqual(0o600, path.stat().st_mode & 0o777)
                self.assertNotIn(b"@example.invalid", path.read_bytes())
            records = json.loads((destination / "records.json").read_text())
            self.assertEqual("local_read_only_recheck", records["mode"])
            self.assertEqual(check.digest(Path(check.__file__).read_bytes()), records["checker_sha256"])
            self.assertEqual(check.digest(workflow.read_bytes()), records["workflow_sha256"])
            self.assertEqual({"records.json", "config.json", "CLA.md", "CLA-PRIVACY.md", "SHA256SUMS"},
                             {path.name for path in destination.iterdir()})

    def test_export_is_refused_in_actions_or_any_git_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / ".git").mkdir()
            with self.assertRaisesRegex(check.Hold, "unsafe_export_location"):
                check.write_export(base / "private-export", self.checker, base / "absent-config.json")
        with mock.patch.dict(os.environ, {"GITHUB_ACTIONS": "true"}):
            with self.assertRaisesRegex(check.Hold, "unsafe_export_location"):
                check.write_export("/tmp/synthetic-never-written-export", self.checker, "absent-config.json")

    def test_untrusted_comment_text_is_data_and_never_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            self.api.comments[0]["body"] += f"\n$(touch {marker}); __import__('os').system('touch {marker}')"
            with mock.patch("os.system", side_effect=AssertionError("Code execution")), \
                    mock.patch("subprocess.run", side_effect=AssertionError("Code execution")):
                self.assert_failed("missing_valid_confirmation")
            self.assertFalse(marker.exists())


class ActivationTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.c = config()
        self.api = FakeApi(self.c)
        self.checker = check.Checker(self.c, self.api, "https://github.com/maredengb/coffered/actions/runs/123")

    def held(self):
        with self.assertRaises(check.Hold):
            self.checker.run()
        self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])

    def test_post_merge_announcement_opens_confirmations(self):
        self.assertEqual("success", self.checker.run()[0]["result"])
        self.assertEqual(920000, self.checker.activation_record["id"])
        self.assertEqual(BASE, self.checker.activation_record["merge_sha"])

    def test_missing_announcement_holds_known_heads_pending(self):
        self.api.activation = None
        self.held()
        self.assertEqual("pending", self.api.statuses[-1][1]["state"])

    def test_nonowner_and_bot_announcements_do_not_open(self):
        for actor in (ALICE, {**OWNER, "type": "Bot"}):
            with self.subTest(actor=actor):
                self.api.activation["user"] = actor
                self.held()

    def test_edited_or_altered_announcement_does_not_open(self):
        original = copy.deepcopy(self.api.activation)
        for field, value in (("updated_at", WHEN), ("body", original["body"] + " extra")):
            with self.subTest(field=field):
                self.api.activation = {**original, field: value}
                self.held()

    def test_unmerged_wrong_owner_wrong_target_or_wrong_sha_is_rejected(self):
        original = copy.deepcopy(self.api.activation_pr)
        for field, value in (("merged", False), ("merged_by", ALICE),
                             ("merge_commit_sha", SHA2), ("number", 98),
                             ("base", {"ref": "other", "repo": {"id": REPOSITORY_ID}}),
                             ("base", {"ref": "main", "repo": {"id": 1}})):
            with self.subTest(field=field, value=value):
                self.api.activation_pr = {**original, field: value}
                self.held()

    def test_announcement_before_or_same_second_as_merge_is_rejected(self):
        for merged_at in (OPENED, WHEN):
            with self.subTest(merged_at=merged_at):
                self.api.activation_pr["merged_at"] = merged_at
                self.held()

    def test_merge_must_belong_to_main(self):
        for ancestry in ({"status": "diverged", "merge_base_commit": {"sha": BASE}},
                         {"status": "ahead", "merge_base_commit": {"sha": SHA2}}):
            with self.subTest(ancestry=ancestry):
                self.api.activation_ancestry = ancestry
                self.held()

    def test_configuration_at_merge_must_match(self):
        self.api.activation_config["enabled"] = False
        self.held()

    def test_activation_config_types_and_encoding_must_match(self):
        self.api.activation_config["schema_version"] = True
        self.held()
        self.api.responses[("GET", PREFIX + "/contents/.github/cla/config.json?ref=" + BASE)] = [
            {"type": "file", "path": ".github/cla/config.json", "encoding": "base64", "content": "%%%"}]
        self.held()

    def test_preopening_and_same_second_confirmations_do_not_pass(self):
        self.api.pulls[1]["created_at"] = "2023-12-28T00:00:00Z"
        for created in ("2023-12-29T12:00:00Z", OPENED):
            with self.subTest(created=created):
                self.api.comments[0].update(created_at=created, updated_at=created)
                self.assertEqual("failure", self.checker.run()[0]["result"])
                self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])

    def test_postopening_confirmation_is_required_for_owner_too(self):
        self.api.pulls[1]["created_at"] = "2023-12-28T00:00:00Z"
        self.api.graphql[SHA1] = author_data(SHA1, actors=(OWNER,))
        self.api.comments = [comment(self.c, actor=OWNER, owner=True)]
        self.api.comments[0].update(created_at=OPENED, updated_at=OPENED)
        self.assertEqual("failure", self.checker.run()[0]["result"])

    def test_announcement_removed_or_edited_before_success_fails(self):
        for fresh in (None, {**self.api.activation, "updated_at": WHEN}):
            with self.subTest(fresh=fresh):
                self.api.fresh_comments[920000] = fresh
                self.assertEqual("failure", self.checker.run()[0]["result"])
                self.assertNotIn("success", [body["state"] for _, body in self.api.statuses])

    def test_activation_api_failure_never_falls_through(self):
        self.api.failures[("GET", PREFIX + "/pulls/99")] = "api_request_failed"
        self.held()

    def test_old_announcement_cannot_activate_changed_configuration(self):
        self.c["recipient_reference_time"] = "2020-01-01T01:00:00Z"
        self.held()

    def test_printing_announcement_never_posts(self):
        with mock.patch.object(check, "load_config", return_value=self.c), \
                mock.patch.object(check, "Api") as api, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, check.main(["--activation-notice", "99", BASE]))
        api.assert_not_called()
        self.assertEqual(check.activation_notice(self.c, 99, BASE), output.getvalue().strip())


class ConfigurationAndEntrypointTests(OfflineTestCase):
    def config_files(self, directory):
        root = Path(directory)
        c = config()
        for key, name in (("agreement", "CLA.md"), ("notice", "CLA-PRIVACY.md")):
            data = ("Synthetic active document: " + name + "\n").encode()
            (root / name).write_bytes(data)
            c[key]["sha256"] = check.digest(data)
        path = root / "config.json"
        path.write_text(json.dumps(c))
        return root, path, c

    def test_config_checks_actual_document_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, c = self.config_files(directory)
            self.assertEqual(c, check.load_config(path, root))
            (root / "CLA.md").write_text("Different bytes")
            with self.assertRaisesRegex(check.Hold, "document_hash_mismatch"):
                check.load_config(path, root)

    def test_enabled_config_rejects_draft_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, c = self.config_files(directory)
            data = b"DRAFT FOR REVIEW"
            (root / "CLA.md").write_bytes(data)
            c["agreement"]["sha256"] = check.digest(data)
            path.write_text(json.dumps(c))
            with self.assertRaisesRegex(check.Hold, "document_not_active"):
                check.load_config(path, root)

    def test_enabled_config_rejects_public_inactive_notices(self):
        documents = (
            ("agreement", "CLA.md",
             b"# Contributor License Agreement\n\n**This CLA is not yet in effect. "
             b"Do not submit a confirmation until activation.**\n"),
            ("notice", "CLA-PRIVACY.md",
             b"# CLA confirmation data\n\n**Online CLA confirmations are not enabled. "
             b"Do not post a confirmation until activation.**\n"),
        )
        for key, name, data in documents:
            with self.subTest(document=name), tempfile.TemporaryDirectory() as directory:
                root, path, c = self.config_files(directory)
                (root / name).write_bytes(data)
                c[key]["sha256"] = check.digest(data)
                path.write_text(json.dumps(c))
                with self.assertRaisesRegex(check.Hold, "document_not_active"):
                    check.load_config(path, root)
                c["enabled"] = False
                path.write_text(json.dumps(c))
                self.assertEqual(c, check.load_config(path, root))

    def test_activation_requires_privacy_and_private_records_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, c = self.config_files(directory)
            for key in ("privacy_ready", "owner_records_ready"):
                with self.subTest(key=key):
                    invalid = copy.deepcopy(c)
                    invalid[key] = False
                    path.write_text(json.dumps(invalid))
                    with self.assertRaisesRegex(check.Hold, "activation_prerequisites_missing"):
                        check.load_config(path, root)

    def test_document_urls_must_be_fixed_repository_commit_urls(self):
        with tempfile.TemporaryDirectory() as directory:
            root, path, c = self.config_files(directory)
            for url in ("https://example.invalid/CLA.md", f"https://github.com/{REPO}/blob/main/CLA.md"):
                with self.subTest(url=url):
                    c["agreement"]["url"] = url
                    path.write_text(json.dumps(c))
                    with self.assertRaisesRegex(check.Hold, "invalid_document_url"):
                        check.load_config(path, root)

    def test_inactive_entrypoint_has_no_api_calls_even_with_token(self):
        c = config()
        c["enabled"] = False
        with mock.patch.object(check, "load_config", return_value=c), \
                mock.patch.object(check, "Api") as api, \
                mock.patch.dict(os.environ, {"GH_TOKEN": "synthetic-sensitive-token"}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, check.main([]))
        api.assert_not_called()
        self.assertIn("PREPARED_NOT_ACTIVE", output.getvalue())
        self.assertNotIn("synthetic-sensitive-token", output.getvalue())

    def test_printing_confirmation_never_posts_or_signs(self):
        c = config()
        with mock.patch.object(check, "load_config", return_value=c), \
                mock.patch.object(check, "Api") as api, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, check.main(["--confirmation", "1", SHA1]))
        api.assert_not_called()
        self.assertEqual(check.confirmation(c, 1, SHA1), output.getvalue().strip())

    def test_local_status_writes_are_refused(self):
        with mock.patch.object(check, "load_config", return_value=config()), \
                mock.patch.object(check, "Api") as api, \
                mock.patch.dict(os.environ, {}, clear=True), \
                contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(1, check.main([]))
        api.assert_not_called()
        self.assertIn("local_status_writes_refused", output.getvalue())

    def run_action(self, event, environment=None, comments=None):
        c = config()
        api = FakeApi(c)
        if comments is not None:
            api.comments = copy.deepcopy(comments)
        with tempfile.TemporaryDirectory() as directory:
            event_path = Path(directory) / "event.json"
            event_path.write_text(json.dumps(event))
            env = {"GITHUB_ACTIONS": "true", "GITHUB_EVENT_PATH": str(event_path),
                   "GITHUB_EVENT_NAME": "pull_request_target", "GITHUB_RUN_ID": "123",
                   "GITHUB_WORKFLOW_SHA": BASE, "GITHUB_REF": "refs/heads/main",
                   "GITHUB_TOKEN": "synthetic-token"}
            env.update(environment or {})
            with mock.patch.object(check, "load_config", return_value=c), \
                    mock.patch.object(check, "Api", return_value=api) as factory, \
                    mock.patch.dict(os.environ, env, clear=True), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                result = check.main([])
        return result, api, factory

    def test_trusted_actions_event_writes_only_statuses_on_pr_head(self):
        result, api, factory = self.run_action({"repository": {"id": REPOSITORY_ID}})
        self.assertEqual(0, result)
        factory.assert_called_once_with("synthetic-token")
        self.assertEqual(SHA1, api.statuses[-1][0])
        writes = [path for method, path in api.calls if method == "POST" and path != "/graphql"]
        self.assertTrue(writes)
        self.assertTrue(all(path == PREFIX + "/statuses/" + SHA1 for path in writes))

    def test_wrong_repository_and_untrusted_workflow_refs_have_no_api_calls(self):
        for event, environment in (({"repository": {"id": REPOSITORY_ID + 1}}, {}),
                                   ({"repository": {"id": REPOSITORY_ID}}, {"GITHUB_REF": "refs/pull/1/merge"}),
                                   ({"repository": {"id": REPOSITORY_ID}}, {"GITHUB_WORKFLOW_SHA": "main"}),
                                   ({"repository": {"id": REPOSITORY_ID}}, {"GITHUB_EVENT_NAME": "pull_request"})):
            with self.subTest(event=event, environment=environment):
                result, api, factory = self.run_action(event, environment)
                self.assertEqual(1, result)
                factory.assert_not_called()
                self.assertEqual([], api.calls)

    def test_unrelated_issue_comment_rechecks_ledger_without_becoming_a_signature(self):
        result, api, factory = self.run_action(
            {"repository": {"id": REPOSITORY_ID}, "issue": {"number": 18}},
            {"GITHUB_EVENT_NAME": "issue_comment"})
        self.assertEqual(0, result)
        factory.assert_called_once_with("synthetic-token")
        self.assertTrue(api.calls)
        self.assertEqual("success", api.statuses[-1][1]["state"])
        self.assertNotIn(("GET", PREFIX + "/issues/18/comments?per_page=100&page=1"), api.calls)

    def test_outside_issue_event_comment_cannot_replace_ledger_confirmation(self):
        result, api, factory = self.run_action(
            {"repository": {"id": REPOSITORY_ID}, "issue": {"number": 18},
             "comment": comment(config())},
            {"GITHUB_EVENT_NAME": "issue_comment"}, comments=[])
        self.assertEqual(1, result)
        factory.assert_called_once_with("synthetic-token")
        self.assertEqual("failure", api.statuses[-1][1]["state"])
        self.assertEqual("missing_valid_confirmation", api.statuses[-1][1]["description"])

    def test_printing_owner_statement_never_posts(self):
        c = config()
        with mock.patch.object(check, "load_config", return_value=c), \
                mock.patch.object(check, "Api") as api, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(0, check.main(["--owner-confirmation", "1", SHA1]))
        api.assert_not_called()
        self.assertEqual(check.owner_confirmation(c, 1, SHA1), output.getvalue().strip())

    def test_logs_exclude_comment_bodies_emails_and_tokens(self):
        c = config()
        api = FakeApi(c)
        api.comments[0]["body"] += "\nsynthetic-private-note"
        with mock.patch.object(check, "load_config", return_value=c), \
                mock.patch.object(check, "Api", return_value=api), \
                mock.patch.dict(os.environ, {"GH_TOKEN": "synthetic-sensitive-token"}, clear=True), \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                contextlib.redirect_stderr(io.StringIO()) as errors:
            self.assertEqual(1, check.main(["--dry-run"]))
        text = output.getvalue() + errors.getvalue()
        for secret in ("synthetic-private-note", "synthetic-sensitive-token", "@example.invalid"):
            self.assertNotIn(secret, text)
        self.assertIn("missing_valid_confirmation", text)


class ApiBoundaryTests(OfflineTestCase):
    def setUp(self):
        super().setUp()
        self.api = check.Api("synthetic-sensitive-token")
        self.api.opener = mock.Mock()

    def test_merge_content_comment_and_foreign_writes_are_refused(self):
        paths = [PREFIX + "/pulls/1/merge", PREFIX + "/contents/README.md", PREFIX + "/issues/17/comments",
                 "/repos/another/project/statuses/" + SHA1, "https://example.invalid/", "//example.invalid/",
                 PREFIX + "/statuses/" + SHA1 + "\n"]
        for path in paths:
            with self.subTest(path=path), self.assertRaises(check.Hold):
                self.api.post(path, {})
        self.api.opener.open.assert_not_called()

    def test_api_errors_are_redacted(self):
        self.api.opener.open.side_effect = urllib.error.URLError("synthetic-sensitive-response")
        with self.assertRaisesRegex(check.Hold, "^api_request_failed$"):
            self.api.get(PREFIX)

    def test_redirects_are_refused(self):
        with self.assertRaisesRegex(check.Hold, "api_redirect_refused"):
            check.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid/")

    def test_request_budget_stops_before_network(self):
        self.api.requests = check.MAX_REQUESTS
        with self.assertRaisesRegex(check.Hold, "api_budget_exhausted"):
            self.api.get(PREFIX)
        self.api.opener.open.assert_not_called()

    def test_partial_graphql_errors_are_not_treated_as_complete_data(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "data": {"repository": {}}, "errors": [{"message": "synthetic-sensitive-response"}]}).encode()
        self.api.opener.open.return_value = response
        with self.assertRaisesRegex(check.Hold, "graphql_incomplete"):
            self.api.post("/graphql", {})


if __name__ == "__main__":
    unittest.main()
