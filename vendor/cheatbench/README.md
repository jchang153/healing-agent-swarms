# CheatBench files (vendored)

Copied from https://github.com/centerforaisafety/cheatbench at the commit in
`SOURCE_COMMIT` (MIT, see `LICENSE`). The first lines of each copied Python file
that was cut from a larger upstream file states which file and lines it comes from.

| File | Upstream | Used for |
|---|---|---|
| `Dockerfile` | `Dockerfile` (verbatim) | the agents' sandbox image, tag `cheatbench-env` |
| `judge.md` | the judge prompt (verbatim) | `experiments/cheatbench_judge.py` |
| `<task>_judge_schema.py` | the task's judge schema (verbatim) | verdict schema and task context for the judge |
| `<task>_verifier.py` | first part of `tasks/<task>/grader.py` (verbatim) | the `verify_factors` / `verify_subset` tool, run on the host |

Not upstream: `fetch.sh` (rebuilds the task data with CheatBench's own builder) and this README.

Build the image (no build arguments, no credentials):

```sh
docker build -t cheatbench-env - < vendor/cheatbench/Dockerfile   # no build context: no repository file is sent to the build
```

Build the task data (git-ignored, written to `data/cheatbench/<task>/`):

```sh
vendor/cheatbench/fetch.sh prime_factorization
vendor/cheatbench/fetch.sh subset_sum
```
