# Coffee transport tasks

Temporary Task 000001 and 000002 share one object layout and episode-local HOME
contract. The numeric IDs can be renamed later without flattening the package.

## Ownership

| File | Owns |
| --- | --- |
| `coffee_transport_common.py` | measured geometry, assets, reset sampling, final success |
| `generation_contract.py` | can order, arm assignment, subtask phases, retarget frames |
| `task_000001/spec.py` | vertical task identity and instruction |
| `task_000001/env_cfg.py` | vertical scene/reset/termination wiring |
| `task_000002/spec.py` | horizontal task identity and instruction |
| `task_000002/env_cfg.py` | horizontal scene/reset/termination wiring |

Future runtime boundary metrics should go in a shared `generation_terms.py` only
when both tasks use the exact same measurement. Task-specific profiles and generation
environments stay below each `task_*/` directory. Dataset campaign counts are not runtime
configuration and belong in experiment manifests, not env classes.

## Fixed behavior

- Can order: right, center, left.
- Active arm: right, right, left.
- Head: default and held.
- Task 000001: lift starts in [-0.28, 0] m, normalizes to 0, transports by lift.
- Task 000002: start is cached HOME plus world +Y offset, normalizes to that same HOME,
  and transports by closed-loop world-pose base control.
