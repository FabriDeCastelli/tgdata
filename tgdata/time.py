from __future__ import annotations

import re

SECONDS = {"s": 1, "min": 60, "h": 3600, "D": 86400, "W": 7 * 86400}
CALENDAR = ("M", "Y")


def to_steps(value: int | str, freq: str) -> int:
    """A step count, or a duration such as "1h" that must be a whole number of `freq` steps."""
    if isinstance(value, int):
        return value
    (n, unit), (step_n, step_unit) = _parse(value), _parse(freq)
    if unit in CALENDAR or step_unit in CALENDAR:
        if unit != step_unit:
            raise ValueError(f"cannot express {value!r} in steps of {freq!r}")
        steps, remainder = divmod(n, step_n)
    else:
        steps, remainder = divmod(n * SECONDS[unit], step_n * SECONDS[step_unit])
    if remainder or not steps:
        raise ValueError(f"{value!r} is not a whole number of {freq!r} steps")
    return steps


def _parse(duration: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d*)\s*(s|min|h|D|W|M|Y)", duration)
    if match is None:
        raise ValueError(f"unrecognised duration {duration!r}")
    return int(match[1] or 1), match[2]
