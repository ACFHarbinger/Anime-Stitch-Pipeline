from __future__ import annotations

import json

from ..other.session import EvaluationSession


def test_load_evaluation_file_switches_output_and_resumes_unrated(tmp_path):
    path = tmp_path / "asp_evaluations_previous.json"
    path.write_text(
        json.dumps({"asp_test01": {"asp": 4, "simple": 3, "reviewed": True}}),
        encoding="utf-8",
    )

    session = EvaluationSession(["asp_test01", "asp_test02"], str(tmp_path / "today.json"))
    session.load_evaluation_file(str(path))

    assert session.out_path == str(path)
    assert session.is_rated("asp_test01")
    assert session.current == "asp_test02"
