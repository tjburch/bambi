"""Cheap checks for manual batch selection and downloaded evidence preservation."""

import json

import pytest

from docs.examples.group_covariance_validation.merge_sbc import merge_campaigns
from docs.examples.group_covariance_validation.workflow_cases import select_cases


@pytest.mark.parametrize(
    "suite,total",
    [
        ("references", 8),
        ("compile-smoke", 8),
        ("single-block", 10),
        ("sbc-pilot", 60),
        ("sbc-full", 600),
        ("samplers", 4),
        ("benchmarks", 1),
        ("heldout", 6),
    ],
)
def test_manual_campaign_batches_cover_every_case_once(suite, total):
    cases = []
    for start in range(0, total, 10):
        cases.extend(select_cases(suite, start, 10)["include"])
    assert len(cases) == total
    assert len({case["id"] for case in cases}) == total
    with pytest.raises(ValueError):
        select_cases(suite, total, 10)


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (0, 11), (True, 1), (0, True)])
def test_invalid_batch_rejected(start, count):
    with pytest.raises(ValueError):
        select_cases("sbc-full", start, count)


def artifact(tmp_path, name, case="a", revision="first"):
    root = tmp_path / name
    entry = root / "cases" / case
    entry.mkdir(parents=True)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "cases": [{"id": "a"}, {"id": "b"}],
            }
        )
    )
    (entry / "status.json").write_text('{"status":"failed"}')
    return root


def test_merge_keeps_failed_cases(tmp_path):
    first, second = artifact(tmp_path, "one"), artifact(tmp_path, "two", "b")
    output = tmp_path / "merged"
    assert merge_campaigns([first, second], output) == 2
    assert json.loads((output / "cases/a/status.json").read_text())["status"] == "failed"
    assert (first / "cases/a/status.json").exists()


@pytest.mark.parametrize("duplicate", [True, False])
def test_merge_rejects_duplicate_or_different_campaign(tmp_path, duplicate):
    first = artifact(tmp_path, "one")
    second = artifact(
        tmp_path, "two", "a" if duplicate else "b", "first" if duplicate else "changed"
    )
    output = tmp_path / "merged"
    with pytest.raises(ValueError):
        merge_campaigns([first, second], output)
    assert not output.exists()
