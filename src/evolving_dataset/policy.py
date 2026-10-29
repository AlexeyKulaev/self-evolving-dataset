"""Numerical scheduling policy, independent from the teacher and student."""
import math
import statistics


def allocate(total, weights, rng):
    if total < 0 or not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Invalid allocation")
    mass = sum(weights.values())
    if mass <= 0:
        raise ValueError("Allocation has no positive mass")
    exact = {k: total * v / mass for k, v in weights.items()}
    result = {k: int(v) for k, v in exact.items()}
    order = list(weights)
    rng.shuffle(order)  # no permanent starvation on equal remainders
    order.sort(key=lambda k: exact[k] - result[k], reverse=True)
    for k in order[:total - sum(result.values())]:
        result[k] += 1
    return result


def progress_weights(before, after):
    # log-space ratio avoids overflow and provides a defined zero-loss limit.
    logs = {}
    for key in before:
        b, a = before[key], after[key]
        if not math.isfinite(b) or not math.isfinite(a) or b < 0 or a < 0:
            raise ValueError("Validation loss must be finite and nonnegative")
        logs[key] = math.log(max(b, 1e-12)) - math.log(max(a, 1e-12))
    top = max(logs.values())
    raw = {k: math.exp(v - top) for k, v in logs.items()}
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def active_quotas(n, active, newcomers, weights, rng):
    old = [k for k in active if k not in newcomers]
    new = [k for k in active if k in newcomers]
    if not old or not new:
        return allocate(n, {k: (weights.get(k, 1) if old else 1) for k in active}, rng)
    new_total = n // 2
    return {**allocate(new_total, dict.fromkeys(new, 1), rng),
            **allocate(n - new_total, {k: weights[k] for k in old}, rng)}


def window(pool, count, accuracy, examples):
    if not 0 <= accuracy <= 1 or count < 0 or count > len(pool):
        raise ValueError("Invalid window request")
    ordered = sorted(pool, key=lambda x: (-examples[x]["c"], x))
    start = min(int(accuracy * len(ordered)), len(ordered) - count)
    return ordered[start:start + count]


def delta(before, after, fixed=None, floor=0.01):
    if fixed is not None:
        return fixed
    if len(before) != len(after) or not before:
        raise ValueError("Paired measurements must have matching nonempty validation sets")
    differences = [int(b) - int(a) for b, a in zip(before, after)]
    se = statistics.stdev(differences) / math.sqrt(len(differences)) if len(differences) > 1 else 0
    return max(floor, 2 * se)


def replay_quotas(n, recovering, rng):
    if not recovering:
        return {}
    drops = {k: value[0] for k, value in recovering.items()}
    fraction = 0.2 if len(drops) > 1 or any(d > 2 * threshold for d, threshold in recovering.values()) else 0.1
    return allocate(int(n * fraction), drops, rng)


def angle_concentration(vectors):
    """Mean cosine over all distinct question/table token pairs, O(tokens * hidden).

    Equivalent to (||sum(normalized h)||^2 - n)/(n*(n-1)). No answer,
    prompt prefix, padding or special tokens are passed by the adapters.
    """
    if len(vectors) == 0:
        raise ValueError("C needs at least one input token")
    normalized = []
    for vector in vectors:
        if len(vector) == 0 or any(not math.isfinite(float(v)) for v in vector):
            raise ValueError("Hidden vectors must be finite and nonempty")
        length = math.sqrt(sum(float(v) ** 2 for v in vector))
        if length == 0:
            raise ValueError("Zero hidden vector has no cosine similarity")
        normalized.append([float(v) / length for v in vector])
    if len({len(v) for v in normalized}) != 1:
        raise ValueError("Hidden dimensions do not match")
    n = len(normalized)
    if n == 1:
        return 0.0  # no distinct pairs; explicit neutral convention
    sums = [sum(v[i] for v in normalized) for i in range(len(normalized[0]))]
    return max(-1.0, min(1.0, (sum(v*v for v in sums) - n) / (n*(n-1))))
