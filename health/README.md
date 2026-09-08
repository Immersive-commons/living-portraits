# health/ — what "this installation is fine" means, mechanically

Nine detectors. Every one exists because something went wrong and **a human had to notice**.

That is the point. On 2026-08-10 a single afternoon's audit found that production had been
running for months without one of its two character specs, that a character had been looping
three clips for an hour, and that one pose had been unreachable for its entire 76-day life.
None of those were subtle. All of them were invisible, because nobody had written down what
"fine" looks like.

```
python main.py verify check projects/living-portraits/health/oracle.yaml
python main.py verify probe projects/living-portraits/health/oracle.yaml   # determinism
python health/checks.py --all                                             # ad-hoc, verbose
```

## The contract

1. **A detector is deterministic.** The same world gives the same verdict twice. `verify
   probe` runs everything twice and calls a flapping check invalid. A health check that
   flaps is not a health check — it is a pager that teaches you to ignore it. So thresholds
   are coarse and every count is read over a fixed window.
2. **A detector never repairs.** Detection and repair stay separate, so a repair can never
   quietly redefine health. Repair belongs to whatever runs the oracle, under its own policy.
3. **Cannot-see is FAIL, never PASS.** If the host is unreachable the check fails with that
   as its stated reason. A checker that reports green when blind is worse than no checker.
4. **A verdict cites its evidence.** Every check prints the numbers it judged on, so a FAIL
   is actionable without a second investigation.
5. **Off-by-choice is a PASS.** `stop_portraits.ps1` turns the show off by DISABLING the
   tasks, and the watchdog respects that so a deliberate stop stays stopped. `panels_alive`
   and `reflection_fired` read the off switch before anything else. The first night this
   suite ran, it did not: someone pressed Stop at 19:55 to use the machine, and the oracle
   called a human's decision a failure every two hours until morning. **A monitor that pages
   you for your own decision is worse than no monitor.** Cannot-*tell* still fails safe — an
   unreadable task state is treated as ON, so a genuinely dark wall is never excused.

## The eight, and the incident behind each

| check | the incident |
|---|---|
| `panels_alive` | the player clean-exits when a fullscreen game steals the display; process liveness was never the question, a live process printing nothing is a dark wall |
| `walker_moving` | Phineas looped three clips for an hour at a pose with one exit |
| `goal_reachable` | the goal menu was built over all edges while the daytime walk masks the bedtime chain — the brain wanted what the body could not reach, for 76 days |
| `lived_integrity` | a flush erased the backfill's provenance while keeping its numbers |
| `deploy_drift` | `maxx.json` missing from production for months |
| `error_rate` | 165 gateway 401s and 62 unparseable replies, unnoticed |
| `world_context` | the IC token went 401 on 2026-07-14 and nothing noticed for 27 days — `context_line()` is fail-soft by contract, so every tick looked healthy while the characters were told nothing about the world |
| `reflection_fired` | reflection runs once a night in a window nobody watches; if it stopped, the only symptom would be a character that slowly stops having opinions |

## Adding one

**When an incident happens that no check caught, that is the moment to add a check** — not
later, and not a general-purpose one. Write the narrowest detector that would have caught
*this* instance, add it to `oracle.yaml` with the incident in a comment, and run `verify
probe` to prove it does not flap.

Resist the urge to write detectors for things that have never happened. Eight real ones beat
thirty imagined ones, and the imagined ones are the ones that flap.

## What these deliberately do NOT check

- **Whether the art is any good.** Nothing here has an opinion about the walk being
  interesting. That is a human judgement and pretending otherwise would make the suite
  unfalsifiable.
- **Whether the graph is well-shaped.** 47% of Phineas's poses have one exit or none. That is
  a real defect, it is measured in `_audit/`, and it is a *design* problem — a health check
  that failed on it would be red forever, which is the same as being off.
- **Anything an LLM would have to judge.** Every verdict here is arithmetic.
