# pumpfun-top5-forecast-01

A deterministic Python 3.13 entry that forecasts the five largest new pump.fun
coins by market cap across the task's answer reads. It ranks coins from the
latest snapshot using log market cap, 30-minute and one-hour momentum, and a
small completed-pool bonus. Per-place softmax probabilities retain 35% unnamed
mass for post-cut newcomers; size probabilities use a horizon-widened normal
model in log10 SOL market-cap bins.

Run locally with the supplied data:

```sh
MAKEFUN_INPUT_DIR=/path/to/input MAKEFUN_OUTPUT_DIR=/tmp/forecast python3 main.py
```

The program reads `task.json` and `history.ndjson`, then writes
`$MAKEFUN_OUTPUT_DIR/forecast.json` (default `/output/forecast.json`). It uses
only the Python standard library and is deterministic.
