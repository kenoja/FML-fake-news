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


def test_api_billing_strips_the_cloud_session_so_the_key_pays(monkeypatch):
    # If any session variable survived, the child call would count against the Claude plan instead.
    monkeypatch.setenv("CLAUDE_CODE_SESSION_ID", "x")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://session-proxy")
    monkeypatch.setenv("AGENT_API_KEY", "sk-test")
    env = tools.child_env({"mode": "api", "api_key_env": "AGENT_API_KEY"})
    assert env["ANTHROPIC_API_KEY"] == "sk-test"
    assert not [k for k in env if k.startswith("CLAUDE") or (k.startswith("ANTHROPIC") and k != "ANTHROPIC_API_KEY")]
    monkeypatch.delenv("AGENT_API_KEY")
    with pytest.raises(RuntimeError):                     # no silent fallback to the plan
        tools.child_env({"mode": "api", "api_key_env": "AGENT_API_KEY"})


def test_spending_cap_pauses_the_cycle_and_raising_it_continues(sandbox, monkeypatch):
    # The owner sets a budget; the agent must stop there instead of spending more.
    from agent import fake_llm
    real_respond, real_config = fake_llm.respond, tools.load_config
    cap = {"usd": 8.0}

    def costly(task_prompt, schema):
        res = real_respond(task_prompt, schema)
        res.cost_usd = 5.0
        return res

    def config(name):
        cfg = real_config(name)
        if name == "agent":
            cfg["billing"]["max_cost_usd_per_cycle"] = cap["usd"]
        return cfg

    monkeypatch.setattr(fake_llm, "respond", costly)
    monkeypatch.setattr(tools, "load_config", config)
    assert agent.run(cycle=1, minutes=60, use_git=False) == "PAUSED"   # plan $5, implement $5 -> $10 >= $8
    assert "spending cap" in (sandbox / ".agent_state" / "last_run.md").read_text()
    cap["usd"] = 100.0
    assert agent.run(cycle=1, minutes=60, use_git=False) == "DONE"
