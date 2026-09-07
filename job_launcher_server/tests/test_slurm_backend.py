"""SlurmBackend tests -- all Slurm CLIs are mocked."""

from __future__ import annotations

from job_launcher_server.registry import PolicyRegistry
from job_launcher_server.slurm_backend import SlurmBackend, SlurmError

from .conftest import make_config


def _backend(tmp_path, fake_slurm, **cfg):
    config = make_config(tmp_path, **cfg)
    registry = PolicyRegistry(config)
    return config, SlurmBackend(config, registry, user="testuser", runner=fake_slurm.run)


def test_submit_parses_parsable_output(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    entry = config.policies["molmoact2-droid"]
    job_id = backend.submit(entry, None)
    assert job_id == "825501"
    # list-form, no shell, uses --parsable and --export=NONE
    cmd = fake_slurm.submitted[0]
    assert cmd[0] == "sbatch"
    assert "--parsable" in cmd
    assert "--export=NONE" in cmd
    assert cmd[-1] == "slurm/molmoact2.sh"


def test_submit_stamps_request_id_as_comment(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    entry = config.policies["molmoact2-droid"]
    backend.submit(entry, "match-123")
    cmd = fake_slurm.submitted[0]
    assert "--comment" in cmd
    assert cmd[cmd.index("--comment") + 1] == "match-123"


def test_submit_failure_raises_sanitized(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    fake_slurm.sbatch_fails = True
    entry = config.policies["molmoact2-droid"]
    try:
        backend.submit(entry, None)
    except SlurmError as exc:
        # message must not contain raw stderr
        assert "something failed" not in str(exc)
    else:
        raise AssertionError("expected SlurmError")


def test_list_jobs_only_returns_owned_policy_jobs(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    fake_slurm.add_job(name="colosseum-molmo", user="testuser")
    fake_slurm.add_job(name="some-other-job", user="testuser")  # not a policy
    fake_slurm.add_job(name="colosseum-molmo", user="someoneelse")  # not ours
    jobs = backend.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].model == "molmoact2-droid"


def test_count_active_model_jobs(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    fake_slurm.add_job(name="colosseum-molmo")
    fake_slurm.add_job(name="colosseum-molmo")
    assert backend.count_active_jobs() == 2
    assert backend.count_active_model_jobs("molmoact2-droid") == 2


def test_get_job_falls_back_to_sacct(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    # add a job, then remove it from the active list to force sacct usage
    jid = fake_slurm.add_job(name="colosseum-molmo", state="COMPLETED")
    active = list(fake_slurm.jobs)
    fake_slurm.jobs = []
    # sacct in the fake matches against jobs list, so put it back only there
    fake_slurm.jobs = active
    job = backend.get_job(jid)
    assert job is not None
    assert job.state == "COMPLETED"


def test_find_by_request_id(tmp_path, fake_slurm):
    config, backend = _backend(tmp_path, fake_slurm)
    fake_slurm.add_job(name="colosseum-molmo", comment="req-abc")
    found = backend.find_by_request_id("req-abc")
    assert found is not None
    assert found.comment == "req-abc"
    assert backend.find_by_request_id("nope") is None
