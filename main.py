#!/usr/bin/env python3
"""Forecast the next pump.fun top five from snapshot history.

The hello protocol is consumed as a batch of historical orders, with every row
anchored to its observed slot. This deterministic program uses no network,
randomness, or wall clock: it ranks current caps with short-run momentum and
writes one forecast for the configured answer window.
"""

import json
import math
import os
from datetime import datetime, timedelta


INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
NAMED_MASS = 0.65
CANDIDATE_COUNT = 12
TEMPERATURE = 0.22
RANK_SHIFT = 0.35
MICRO = 1_000_000


def parse_time(text):
    """Parse the ISO UTC timestamps used by the task and marks."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def bounded(value, low, high):
    return max(low, min(high, value))


def cap_log(lamports):
    """Return log10 market cap measured in SOL, preserving integer input."""
    return math.log10(int(lamports)) - 9.0


def upper_normal(x, mean, deviation):
    if x == -math.inf:
        return 1.0
    if x == math.inf:
        return 0.0
    return 0.5 * math.erfc((x - mean) / (deviation * math.sqrt(2.0)))


def floor_micro(value):
    """Round down as required while protecting against tiny FP negatives."""
    return math.floor(max(0.0, value) * MICRO) / MICRO


def row_rank(row):
    """A malformed rank sorts last rather than preventing a valid forecast."""
    try:
        return int(row.get("rank"))
    except (TypeError, ValueError):
        return 10**9


def load_marks(filename):
    marks = []
    with open(filename, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            mark = json.loads(line)
            if mark.get("ok") is True and isinstance(mark.get("rows"), list):
                marks.append(mark)
    marks.sort(key=lambda mark: (mark.get("at", ""), str(mark.get("slot", ""))))
    return marks


def previous_cap(observations, cutoff):
    """Latest observed cap at or before cutoff, or None for a new coin."""
    answer = None
    for observed_at, cap in observations:
        if observed_at > cutoff:
            break
        answer = cap
    return answer


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as source:
        task = json.load(source)
    marks = load_marks(os.path.join(INPUT_DIR, "history.ndjson"))
    if not marks:
        raise SystemExit("history has no successful marks")

    places = int(task["places"])
    edges = [float(edge) for edge in task["size"]["edges"]]
    output_name = task.get("output", "forecast.json")
    last_mark = marks[-1]
    last_at = parse_time(last_mark["at"])

    # Record only observations actually present in the historical chart.
    history = {}
    for mark in marks:
        observed_at = parse_time(mark["at"])
        for row in mark["rows"]:
            mint = row.get("mint")
            try:
                cap = int(row.get("market_cap_lamports"))
            except (TypeError, ValueError):
                continue
            if mint and cap > 0:
                history.setdefault(mint, []).append((observed_at, cap))
    for observations in history.values():
        observations.sort(key=lambda point: point[0])

    # Deduplicate defensively by rank then mint so top5 is always distinct.
    last_rows = sorted(last_mark["rows"], key=lambda row: (row_rank(row), row.get("mint", "")))
    seen = set()
    coins = []
    for row in last_rows:
        mint = row.get("mint")
        if not mint or mint in seen or mint not in history:
            continue
        seen.add(mint)
        try:
            current = int(row.get("market_cap_lamports"))
        except (TypeError, ValueError):
            continue
        if current <= 0:
            continue
        observations = history[mint]
        cap_30 = previous_cap(observations, last_at - timedelta(minutes=30))
        cap_60 = previous_cap(observations, last_at - timedelta(minutes=60))
        growth_30 = math.log10(current / cap_30) if cap_30 else 0.0
        growth_60 = math.log10(current / cap_60) if cap_60 else 0.0
        momentum = (bounded(growth_30, -1.0, 1.0) + bounded(growth_60, -1.0, 1.0)) / 2.0
        graduated = 1.0 if row.get("venue") == "pool" and row.get("complete") else 0.0
        current_log = cap_log(current)
        coins.append({
            "mint": mint,
            "score": current_log + 0.7 * momentum + 0.12 * graduated,
            "size_mean": current_log + 0.5 * bounded(growth_60, -1.0, 1.0),
        })

    if len(coins) < places:
        raise SystemExit("last mark has fewer usable coins than requested places")
    coins.sort(key=lambda coin: (-coin["score"], coin["mint"]))
    for ordinal, coin in enumerate(coins, start=1):
        coin["forecast_rank"] = ordinal

    answer_times = task.get("reads", [])
    horizon_hours = 0.0
    if answer_times:
        horizon_hours = max(0.0, (parse_time(answer_times[0]) - last_at).total_seconds() / 3600.0)
    size_sd = min(0.65, 0.35 + 0.03 * horizon_hours)

    candidates = coins[:CANDIDATE_COUNT]
    probabilities = []
    for place in range(1, places + 1):
        shifted = [coin["score"] - RANK_SHIFT * abs(coin["forecast_rank"] - place) for coin in candidates]
        highest = max(shifted)
        weights = [math.exp((value - highest) / TEMPERATURE) for value in shifted]
        denominator = sum(weights)
        for coin, weight in zip(candidates, weights):
            probability = floor_micro(NAMED_MASS * weight / denominator)
            if probability:
                probabilities.append({"mint": coin["mint"], "place": place, "p": probability})

    sizes = []
    for place in range(1, places + 1):
        mean = coins[place - 1]["size_mean"]
        for bin_number in range(len(edges) + 1):
            low = -math.inf if bin_number == 0 else edges[bin_number - 1]
            high = math.inf if bin_number == len(edges) else edges[bin_number]
            probability = floor_micro(upper_normal(low, mean, size_sd) - upper_normal(high, mean, size_sd))
            if probability:
                sizes.append({"place": place, "bin": bin_number, "p": probability})

    output = {"probabilities": probabilities, "top5": [coin["mint"] for coin in coins[:places]], "sizes": sizes}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as target:
        target.write(json.dumps(output, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
