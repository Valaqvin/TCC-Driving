# Closed-Loop CARLA Evaluation

This note describes the reproducible comparison between the baseline
cooperative-driving pipeline and the TTC-calibrated bargaining variant.

## Scope

The intended comparison is:

1. `baseline`: cooperative driving without the bargaining action module.
2. `bargaining`: the same perception and planning stack with Rubinstein
   bargaining enabled.

The two variants should use the same routes, scenarios, random seeds, and
pretrained perception/planning weights. The public repository contains the
launchers and route files, but it does not contain CARLA binaries or model
checkpoints.

## External Resources

Install a compatible CARLA release separately and set its location before
running the evaluator:

```powershell
$env:CARLA_ROOT = "C:\path\to\CARLA"
```

Place compatible checkpoints in the ignored local directory:

```text
ckpt/
└── codriving/
    ├── perception/
    │   ├── config.yaml
    │   └── net_epoch_bestval_at16.pth
    └── planning/
        └── epoch_26.ckpt
```

The exact checkpoint names can be changed in the launcher and agent
configuration files when using another trained model.

## Smoke Run

Start the CARLA server on port `2000`, then run these commands from the
repository root:

```powershell
python server_preflight.py --host 127.0.0.1 --port 2000
python launch_closed_loop_suite.py --variant suite --carla-port 2000
```

To use another port:

```powershell
python launch_closed_loop_suite.py --variant suite --carla-port 2002
```

To inspect the generated command and validate local paths without starting
CARLA:

```powershell
python server_preflight.py --skip-carla
python launch_closed_loop_suite.py --variant suite --carla-port 2000 --dry-run
```

## Result Summaries

After the suite finishes, summarize the evaluator outputs:

```powershell
python summarize_closed_loop_results.py Result\closed_loop_suite
```

This writes:

```text
Result/closed_loop_suite/closed_loop_summary.csv
Result/closed_loop_suite/closed_loop_summary.md
```

For the bargaining variant, also inspect the negotiation log:

```powershell
python summarize_bargain_log.py Result\closed_loop_suite\<bargaining_run>\bargain_log.npy
```

A valid bargaining run should report:

- active bargaining steps greater than zero;
- `input modes` containing `encoded_state`;
- a nonzero maximum waypoint delta;
- a nonempty evaluator `records` field.

If the log contains only skipped entries, inspect the recorded skip reason.
The bargaining wrapper expects the perception model to expose the
`fused_feature` output key.

## Recommended Table

For a preliminary comparison, report:

- driving score;
- route completion;
- infraction penalty;
- vehicle collisions;
- layout collisions;
- red-light infractions;
- route timeout;
- vehicle blocked;
- active bargaining steps;
- mean and maximum waypoint delta.

Run at least three repetitions on the Town10 smoke route and three
repetitions on one non-Town10 route. Use identical routes and seeds for the
baseline and bargaining variants.

Generated results, checkpoints, and local CARLA files are ignored by Git and
should not be committed to the public repository.
