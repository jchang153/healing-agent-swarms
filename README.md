# Healing Agent Swarms

Multi-agent experiments on Conflicting-LiveCodeBench from ImpossibleBench (Zhong, Raghunathan, Carlini, 2025), using the paper's full scaffold through Inspect, Docker sandboxes, and OpenRouter. Each task is solved by N agents, each in its own workspace, with an optional shared message board.

## What you need

- Python 3.12
- Docker
- An OpenRouter API key, only for a paid live run

Set up the runtime once:

```sh
python3.12 -m venv .venv-impossible
.venv-impossible/bin/pip install -r experiments/impossible-runtime-lock.txt
```

The sandbox image (`aisiuk/inspect-tool-support`, the image the upstream code specifies) is pinned by digest in `experiments/lcb_swarm_campaign.py` and pulled on first use. On this Mac, if `docker` is not on PATH, run `export PATH="$HOME/.docker/bin:$PATH"`.

Run the offline unit tests with `.venv-impossible/bin/python -m unittest discover -s tests -v`.

## The experiment

`experiments/lcb_swarm_campaign.py` runs the paper's LiveCodeBench full scaffold: 10 submissions and 50 messages per agent, and the unmodified upstream scorer. `--prompt A|B|C|D` selects the paper's Table 1 prompt variant (default D, the strictest; the paper's Table 1 results used the minimal scaffold). Agents work independently, each in a fresh container with its own `func.py` and `test.py`. In the `board` condition agents can call `board_read` and `board_post`; these tools run on the host. The board is described in a system message at the start, and a one-sentence reminder is appended to every failed-submission feedback message. The board is pull-only: agents see posts only when they call `board_read`. Limits are per agent.

Task data is pinned in `experiments/lcb_data/` (see `PROVENANCE.json`). `experiments/lcb_validate_data.py` mechanically confirms 96 of the 103 conflicting tasks as impossible, lists 6 for manual review, and reports 1 malformed test (`lcbhard_77`). Runs use the confirmed set by default.

Scripted check (no API calls, zero cost):

```sh
.venv-impossible/bin/python experiments/lcb_swarm_campaign.py --mode check
```

A live run needs `--mode live`, `--condition board|independent`, and `--max-usd`. Use `--agents 1 --condition independent` for the single-agent baseline and `--split original` for the capability baseline. `--repeats` adds fresh groups per task, `--limit` restricts the number of tasks, and `--human-intervention` enables the flag option (off by default, as in the paper's main results). Paid runs require explicit authorization.

Agents within a group generate concurrently. `--parallel-groups G` runs G tasks at once in one Inspect call so all their agents start together, and `--max-in-flight K` (default 8) caps concurrent model requests across all agents. The gateway reserves budget atomically before each request and settles it after, so concurrency does not let reservations exceed the allowance. Each agent container is capped at 1 CPU and 1 GiB; size `G × agents` to the Docker host.

## Run output

**Start with `runs/INDEX.md`.** It lists every run oldest first by its recorded start time, with prompt, setup, outcomes, spend, a validity flag (`valid`, `censored`, `aborted`, `test`), and notes on purpose and known problems. `runs/index.json` holds the same data for scripts. Both are regenerated after every run, or manually with `python3 experiments/index_runs.py`. Hand-written notes for each run live in `experiments/run_annotations.json`, which is tracked in Git; `runs/` itself is ignored by Git, so back it up separately.

New runs are written to `runs/<UTC start>_<label>_<id>/`, so an alphabetical listing is chronological; scripted checks go to `runs/checks/`. Pass `--label` for the folder name and `--note "why this run"` to record its purpose. Each manifest records the Git commit and whether the code had uncommitted changes. Runs from before this layout keep their `runs/lcb-<id>/` names; the index orders them correctly.

Each run folder contains a manifest of every setting, per-agent Inspect logs and transcripts, per-scoring submission records with workspace file hashes, board logs, billing records, and `report.md`. Each agent run is labelled `pass_tests_modified`, `pass_tests_intact`, `flagged`, `fail_tests_modified`, `censored` (stopped by a harness limit), or `fail`. The report gives agent-level and group-level pass rates with 95% Wilson intervals; use the group-level rate for the board condition, where agents are not independent.

On impossible tasks, a pass is the paper's cheating measure and a lower bound on cheating. Labels are deterministic and do not establish intent.

## Upstream scoring caveats

The scorer always restores the supplied `test.py` before grading (upstream `record_to_sample` always stores it), so test edits are detected but never help. It passes on exit code 0 alone, so a `func.py` that exits on import scores as a pass (`pass_tests_intact`). Both behaviours are kept to match the reference implementation.

## Spending and credentials

Put `OPENROUTER_API_KEY` in the local `.env` file (ignored by Git, `chmod 600`). The key is read on the host by `healing_swarm/secrets.py` and used only inside a loopback budget gateway; containers never receive it and are started without network access. Never put real keys in `.env.example`, paste them in chat, or print the file.

The gateway reserves a conservative cost before each request and keeps a $0.50 contingency. Accounting happens after each response, so spending is a threshold, not a hard cap; use a dedicated OpenRouter key with its own provider-side limit. A cancelled or failed request may still be billed; the summary marks accounting as uncertain in that case, and the provider activity should be reconciled before restarting.

`vendor/impossiblebench` holds the pinned upstream source and its license.
