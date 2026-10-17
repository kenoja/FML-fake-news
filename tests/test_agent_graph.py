"""The overnight loop must survive being stopped at any point: work done before a stop is never
repeated or lost, and the next run continues from the last finished step."""
import pytest

from agent import agent, memory, nodes, tools


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("FML_AGENT_FAKE_LLM", "1")
    monkeypatch.setattr(memory, "STATE_DIR", tmp_path / ".agent_state")
    monkeypatch.setattr(memory, "REPO", tmp_path)
    # no real pytest / pipeline / site build inside the test (that would recurse)
    monkeypatch.setattr(tools, "run_cmd", lambda *a, **k: (0, "ok"))
    monkeypatch.setattr(nodes, "tools", tools)
    monkeypatch.setattr(tools, "run_gates", lambda *a, **k: [])
    return tmp_path


def history(cycle):
    return [e["node"] for e in agent._final_state(cycle)["history"]]


def test_full_cycle_reaches_done_and_writes_the_ai_log(sandbox):
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
    assert history(1) == ["load_context", "plan", "implement", "run_pipeline", "verify",
                          "build_site", "review", "finalize"]
    assert "Cycle 1" in (sandbox / "docs" / "ai-log.md").read_text()


def test_a_stopped_run_resumes_where_it_left_off(sandbox):
    # minutes=0: stop after every single step, like a run that keeps hitting its time budget
    statuses = []
    for _ in range(20):
        statuses.append(agent.run(cycle=1, minutes=0, use_git=False))
        if statuses[-1] == "DONE":
            break
    assert statuses[-1] == "DONE"
    assert set(statuses[:-1]) == {"CONTINUE"}
    # every node ran exactly once despite seven interruptions
    assert history(1) == ["load_context", "plan", "implement", "run_pipeline", "verify",
                          "build_site", "review", "finalize"]


def test_a_finished_cycle_is_not_run_again(sandbox):
    agent.run(cycle=1, minutes=60, use_git=False)
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
    assert history(1).count("finalize") == 1


def test_usage_limit_pauses_and_keeps_progress(sandbox, monkeypatch):
    calls = {"n": 0}
    real = tools.run_claude

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # plan succeeds, implement hits the limit
            raise tools.UsageLimitReached("usage limit reached")
        return real(*a, **k)

    monkeypatch.setattr(tools, "run_claude", flaky)
    assert agent.run(cycle=1, minutes=60, use_git=False) == "PAUSED"
    assert history(1)[-1] == "plan"
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
    assert history(1).count("plan") == 1


def test_low_review_scores_send_the_work_back_to_planning(sandbox, monkeypatch):
    rounds = {"n": 0}
    from agent import fake_llm
    real = fake_llm.respond

    def strict_reviewer(task_prompt, schema):
        res = real(task_prompt, schema)
        if schema and "scores" in schema.get("properties", {}):
            rounds["n"] += 1
            if rounds["n"] == 1:
                res.structured["scores"][0]["score"] = 7
                res.structured["must_fix"] = [{"criterion": "question", "issue": "x", "where": "y", "fix": "z"}]
        return res

    monkeypatch.setattr(fake_llm, "respond", strict_reviewer)
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
    assert history(1).count("review") == 2
    assert history(1).count("plan") == 2


def test_a_step_that_runs_out_of_time_is_retried_not_failed(sandbox, monkeypatch):
    # A slow task must not end the cycle: its partial work is kept and the same step runs again.
    calls = {"n": 0}
    real = tools.run_claude

    def slow_once(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:  # the first implement call times out
            raise TimeoutError("Claude Code call exceeded 1080s")
        return real(*a, **k)

    monkeypatch.setattr(tools, "run_claude", slow_once)
    assert agent.run(cycle=1, minutes=60, use_git=False) == "CONTINUE"
    assert history(1)[-1] == "plan"
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
    assert history(1).count("implement") == 1          # retried, not skipped or duplicated


def test_the_same_step_timing_out_three_times_fails_loudly(sandbox, monkeypatch):
    # Endless retries would burn usage for nothing; after three in a row a human has to look.
    real = tools.run_claude
    calls = {"n": 0}

    def plan_ok_then_always_slow(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return real(*a, **k)
        raise TimeoutError("too slow")

    monkeypatch.setattr(tools, "run_claude", plan_ok_then_always_slow)
    statuses = [agent.run(cycle=1, minutes=60, use_git=False) for _ in range(3)]
    assert statuses == ["CONTINUE", "CONTINUE", "FAILED"]


def test_restarting_after_a_timeout_failure_gives_fresh_attempts(sandbox, monkeypatch):
    # After a FAILED run the owner fixes something and presses Run now; the step must get new tries.
    real = tools.run_claude
    calls = {"n": 0}

    def plan_ok_then_slow_three_times(*a, **k):
        calls["n"] += 1
        if 2 <= calls["n"] <= 4:
            raise TimeoutError("too slow")
        return real(*a, **k)

    monkeypatch.setattr(tools, "run_claude", plan_ok_then_slow_three_times)
    assert [agent.run(cycle=1, minutes=60, use_git=False) for _ in range(3)] == ["CONTINUE", "CONTINUE", "FAILED"]
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
