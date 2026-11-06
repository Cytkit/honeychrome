"""
covariate_check.py — find and fix problems in a sample table's covariates
========================================================================

Shared by the plugins with a groups-and-covariates table (Automated Gating,
Honeycluster). ``check_covariates`` summarises what would trouble the
statistics, one line per problem and column (never one per cell), each with
a suggested fix; ``apply_fixes`` carries out the fixes the user picked.

Covariates are passed as {column: {sample: value}}, the samples being those
the statistics would use.

What is checked, per column:

* **Blank cells**: fix by filling them with ``NA`` (the statistics treat NA
  as no value).
* **Missing-value spellings** other than ``NA`` (``na``, ``n/a``, ``-``…;
  ``differential_stats.MISSING_WORDS``): fix by changing them to ``NA``.
* **Unknown** and its spellings (``Unknown``, ``Uknown``, ``not known``,
  ``?``…): fix by changing them to ``NA``. Left as they are, they are one
  more category.
* **Near-duplicate categories**: values differing only in case, spacing or
  punctuation, or by one or two letters (``Female``/``female``,
  ``Unknown``/``Uknown``): fix by merging into the most common spelling.
* **Numbers written as text**: when most values read as numbers once
  converted (see ``parse_number``), fix by converting them and entering the
  column as Numeric. Cells that cannot be read are listed for editing by
  hand. A column that is already plain numbers with more than five distinct
  values is offered as Numeric.
* **Confounding**, per comparison (two groups): a covariate whose values
  decide the group within the comparison (every sample of one group shares
  one value, or nearly every sample has its own value) makes the model
  impossible to fit. Missing values are first resolved as the statistics do
  (``differential_stats.comparison_parts``). One line per comparison names
  the covariates; the user edits the values or removes the column. A second
  line per comparison names the covariates the statistics will leave out of
  that comparison because one group has no value.

Every column can also be removed. Fixes are applied in order: removals,
missing values, merges, then conversions, so a conversion sees NA rather than
'na'.
"""

from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from honeychrome.controller_components import differential_stats as ds

NA = 'NA'

UNKNOWN_WORDS = frozenset({'unknown', 'unk', 'unkn', 'not known', 'n/k', 'nk', '?', '??',
                           'not recorded', 'not stated', 'unspecified', 'undetermined'})
_UNKNOWN_SIMILARITY = 0.85

#: A column is offered for conversion when at least this share of its values
#: (not counting missing ones) reads as numbers.
NUMERIC_SHARE = 0.8
#: Plain-number columns with more distinct values than this are offered as
#: Numeric; fewer are probably codes (batch 1, 2, 3).
NUMERIC_MIN_DISTINCT = 5

NO_ACTION = 'none'
FILL_NA = 'fill_na'
TO_NA = 'to_na'
REMOVE = 'remove'
MERGE = 'merge'
CONVERT = 'convert'
TICK_NUMERIC = 'tick_numeric'

ACTION_LABELS = {
    FILL_NA: "Fill blanks with NA",
    TO_NA: "Change to NA",
    MERGE: "Merge spellings",
    CONVERT: "Convert to numbers (Numeric)",
    TICK_NUMERIC: "Tick Numeric",
    REMOVE: "Remove column",
    NO_ACTION: "Take no action",
}
_ORDER = (REMOVE, FILL_NA, TO_NA, MERGE, CONVERT, TICK_NUMERIC)


@dataclass
class Issue:
    """One problem: ``kind``, the ``column`` it concerns (or '' for a
    comparison), a one-line ``summary``, the ``actions`` offered (first =
    suggested) and what a fix needs (``payload``)."""
    kind: str
    column: str
    summary: str
    actions: list[str]
    payload: dict = field(default_factory=dict)
    comparison: str = ''


def _examples(values, n: int = 4) -> str:
    shown = list(dict.fromkeys(values))
    return ', '.join(f"'{v}'" for v in shown[:n]) + (' …' if len(shown) > n else '')


def _is_unknown(text: str) -> bool:
    low = text.strip().lower()
    if low in UNKNOWN_WORDS:
        return True
    return len(low) >= 5 and difflib.SequenceMatcher(None, low, 'unknown').ratio() >= _UNKNOWN_SIMILARITY


def _value_kind(value) -> str:
    """'blank', 'na' (exactly NA), 'missing' (another missing spelling),
    'unknown' or 'value'."""
    if value is None or not str(value).strip():
        return 'blank'
    text = str(value).strip()
    if text == NA:
        return 'na'
    if ds.is_missing_value(text):
        return 'missing'
    if _is_unknown(text):
        return 'unknown'
    return 'value'


# ---------------------------------------------------------------------------
# Numbers written as text
# ---------------------------------------------------------------------------

_COMMENT = re.compile(r'\([^)]*\)|\[[^\]]*\]')
_THOUSANDS = re.compile(r'^[-+]?\d{1,3}(,\d{3})+(\.\d+)?$')
_SCIENTIFIC = re.compile(r'^([-+]?\d*\.?\d+)\s*[x×*]\s*10\s*\^\s*([-+]?\d+)$', re.IGNORECASE)
_WEEKS_DAYS = re.compile(r'^(\d+)\s*\+\s*([0-6])$')


def parse_number(text) -> float | None:
    """A number from text, or None.

    Reads plain numbers, ``98%`` (98), thousands separators (``768,500``),
    powers of ten (``1.116x10^6``, ``5.4 X 10^6``, ``2e6``) and gestation as
    weeks+days (``33+1`` → 33 + 1/7 weeks). Comments in brackets are
    dropped first (``5.8x10^6 (bloody)``). Anything else is None.
    """
    t = _COMMENT.sub('', str(text)).strip()
    if not t:
        return None
    if t.endswith('%'):
        t = t[:-1].strip()
    if _THOUSANDS.match(t):
        t = t.replace(',', '')
    m = _SCIENTIFIC.match(t)
    if m:
        return float(m.group(1)) * 10.0 ** int(m.group(2))
    m = _WEEKS_DAYS.match(t)
    if m:
        return round(int(m.group(1)) + int(m.group(2)) / 7.0, 4)
    try:
        value = float(t)
    except ValueError:
        return None
    return value if value == value and abs(value) != float('inf') else None


def format_number(value: float) -> str:
    """Text for a converted number: whole numbers without a decimal point."""
    return str(int(value)) if float(value).is_integer() else f"{value:.10g}"


def _plain_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Near-duplicate categories
# ---------------------------------------------------------------------------

def _squash(text: str) -> str:
    return re.sub(r'[^a-z0-9]', '', text.lower())


def _similar(a: str, b: str) -> bool:
    sa, sb = _squash(a), _squash(b)
    if sa == sb:
        return True
    if min(len(sa), len(sb)) < 5 or any(ch.isdigit() for ch in sa + sb):
        return False
    return difflib.SequenceMatcher(None, sa, sb).ratio() >= 0.85


def near_duplicates(counts: Counter) -> list[list[str]]:
    """Sets of spellings that look like one category, most common first."""
    values = [v for v, _n in counts.most_common()]
    seen, sets = set(), []
    for v in values:
        if v in seen:
            continue
        family = [v] + [w for w in values if w != v and w not in seen and _similar(v, w)]
        if len(family) > 1:
            sets.append(family)
            seen.update(family)
    return sets


# ---------------------------------------------------------------------------
# The check
# ---------------------------------------------------------------------------

def _column_issues(name: str, values: dict, samples: list[str], numeric: bool) -> list[Issue]:
    issues = []
    cells = {s: values.get(s) for s in samples}
    by_kind: dict[str, list[str]] = {}
    for s, v in cells.items():
        by_kind.setdefault(_value_kind(v), []).append(s)

    blanks = by_kind.get('blank', [])
    if blanks:
        issues.append(Issue('blank', name, f"{len(blanks)} blank cell(s).",
                            [FILL_NA, REMOVE, NO_ACTION]))
    other_na = by_kind.get('missing', [])
    if other_na:
        spellings = [str(cells[s]).strip() for s in other_na]
        issues.append(Issue('missing', name, f"{len(other_na)} cell(s) spell 'no value' as "
                            f"{_examples(spellings)}.", [TO_NA, REMOVE, NO_ACTION],
                            {'values': sorted(set(spellings))}))
    unknown = by_kind.get('unknown', [])
    if unknown:
        spellings = [str(cells[s]).strip() for s in unknown]
        issues.append(Issue('unknown', name, f"{len(unknown)} cell(s) say "
                            f"{_examples(spellings)}, which the statistics count as one more "
                            "category, not as no value.", [TO_NA, REMOVE, NO_ACTION],
                            {'values': sorted(set(spellings))}))

    filled = [str(cells[s]).strip() for s in by_kind.get('value', [])]
    if not filled:
        return issues
    parsed = {v: parse_number(v) for v in set(filled)}
    readable = [v for v in filled if parsed[v] is not None]
    unreadable = [v for v in filled if parsed[v] is None]
    share = len(readable) / len(filled)
    needs_conversion = any(not _plain_number(v) for v in readable)
    distinct = len({parsed[v] for v in readable})

    if numeric:
        if unreadable:
            issues.append(Issue('unreadable', name, f"Numeric, but {len(unreadable)} cell(s) "
                                f"are not numbers: {_examples(unreadable)}. Edit them by hand "
                                "or change them to NA.", [NO_ACTION, TO_NA],
                                {'values': sorted(set(unreadable))}))
        if needs_conversion:
            issues.append(Issue('convert', name, f"Numeric, with {_examples([v for v in readable if not _plain_number(v)])} "
                                "written as text; convert them to plain numbers.",
                                [CONVERT, NO_ACTION]))
        return issues

    if share >= NUMERIC_SHARE and (needs_conversion or distinct > NUMERIC_MIN_DISTINCT):
        if needs_conversion:
            examples = [v for v in readable if not _plain_number(v)][:3]
            shown = ', '.join(f"'{v}' → {format_number(parsed[v])}" for v in examples)
            summary = (f"{len(readable)} of {len(filled)} values read as numbers once "
                       f"converted ({shown}).")
            actions = [CONVERT, REMOVE, NO_ACTION]
        else:
            summary = (f"Every value is a number ({distinct} distinct); enter it as a "
                       "continuous number rather than as categories.")
            actions = [TICK_NUMERIC, NO_ACTION]
        if unreadable:
            summary += (f" {len(unreadable)} cell(s) cannot be read and stay as they are, "
                        f"for editing by hand: {_examples(unreadable)}.")
        issues.append(Issue('convert', name, summary, actions))
        return issues

    for family in near_duplicates(Counter(filled)):
        issues.append(Issue('duplicate', name, f"{_examples(family)} look like one category.",
                            [MERGE, NO_ACTION], {'into': family[0], 'values': family[1:]}))
    return issues


def _confounding_issues(covariates: dict, samples: list[str], groups: dict,
                        comparisons: list[tuple[str, str]], numeric: list[str]) -> list[Issue]:
    names = list(covariates)
    if not names or not comparisons:
        return []
    frame = pd.DataFrame(
        {n: [covariates[n].get(s) for s in samples] for n in names}, index=samples)
    issues = []
    group_vec = [groups.get(s, '') for s in samples]
    for base, other in comparisons:
        label = f"{other} vs {base}"
        rows = [i for i, g in enumerate(group_vec) if g in (base, other)]
        if not rows:
            continue
        sub_samples = [samples[i] for i in rows]
        sub_groups = [group_vec[i] for i in rows]
        cov = ds.covariate_frame(frame, sub_samples, names, numeric=[], allow_missing=True)
        part = ds.comparison_parts(sub_groups, [(base, other)], cov, min_per_group=1)[0]
        kept = list(part['covariates'].columns) if part['covariates'] is not None else []
        left_out = [n for n in names if n not in kept]
        if left_out:
            issues.append(Issue('left_out', '', f"{len(left_out)} covariate(s) have no value in "
                                "one of the groups and will be left out of this comparison's "
                                f"model: {_examples(left_out, 6)}.",
                                [NO_ACTION], comparison=label))
        confounded, unique = [], []
        keep_rows = [i for i, keep in enumerate(part['mask']) if keep]
        for n in kept:
            col = cov.iloc[keep_rows][[n]]
            values = col[n].tolist()
            if n in numeric:
                parsed = [parse_number(v) for v in values]
                if all(v is not None for v in parsed):
                    col = pd.DataFrame({n: parsed}, index=col.index)
            g = [sub_groups[i] for i in keep_rows]
            try:
                ds.build_design(g, base, None, col)
            except RuntimeError:
                (unique if len(set(values)) >= 0.8 * len(values) else confounded).append(n)
        if confounded:
            issues.append(Issue('confounded', '', f"The values of {_examples(confounded, 6)} "
                                "decide the group (e.g. every sample of one group shares one "
                                "value), so the model cannot be fitted with them. Edit those "
                                "values, change them to NA in the group concerned, or leave "
                                "the covariate out of 'Adjust for'.",
                                [NO_ACTION], {'columns': confounded}, comparison=label))
        if unique:
            issues.append(Issue('confounded', '', f"Nearly every sample has its own value of "
                                f"{_examples(unique, 6)}. Use a donor column as "
                                "the pairing variable, convert numbers to Numeric, or leave "
                                "these out of 'Adjust for'.",
                                [NO_ACTION], {'columns': unique}, comparison=label))
    return issues


def check_covariates(covariates: dict, samples: list[str], groups: dict,
                     comparisons: list[tuple[str, str]], numeric: list[str]) -> list[Issue]:
    """Every problem in *covariates* ({column: {sample: value}}) for the
    *samples* the statistics would use, their *groups* ({sample: group}),
    the *comparisons* [(base, other)] and the *numeric* columns. Column
    problems come first, in column order, then one or two lines per
    comparison."""
    issues = []
    for name, values in covariates.items():
        issues.extend(_column_issues(name, values or {}, samples, name in numeric))
    issues.extend(_confounding_issues(covariates, samples, groups, comparisons, numeric))
    return issues


def apply_fixes(covariates: dict, numeric: list[str], samples: list[str],
                fixes: list[tuple[Issue, str]]) -> tuple[dict, list[str], list[str]]:
    """Carry out the chosen fixes. *fixes* pairs each Issue with one of its
    actions. Returns (covariates, numeric columns, removed columns); the
    inputs are not changed. Only *samples* are edited."""
    covs = {n: dict(v or {}) for n, v in covariates.items()}
    numeric = list(numeric)
    removed: list[str] = []
    chosen = [(i, a) for i, a in fixes if a != NO_ACTION]
    for action in _ORDER:
        for issue, picked in chosen:
            if picked != action:
                continue
            name = issue.column
            if not name or name in removed:
                continue
            col = covs.setdefault(name, {})
            if action == REMOVE:
                covs.pop(name, None)
                removed.append(name)
                if name in numeric:
                    numeric.remove(name)
            elif action == FILL_NA:
                for s in samples:
                    if _value_kind(col.get(s)) == 'blank':
                        col[s] = NA
            elif action == TO_NA:
                targets = set(issue.payload.get('values', []))
                for s in samples:
                    if col.get(s) is not None and str(col[s]).strip() in targets:
                        col[s] = NA
            elif action == MERGE:
                targets = set(issue.payload.get('values', []))
                for s in samples:
                    if col.get(s) is not None and str(col[s]).strip() in targets:
                        col[s] = issue.payload['into']
            elif action == CONVERT:
                for s in samples:
                    v = col.get(s)
                    if _value_kind(v) != 'value':
                        continue
                    number = parse_number(v)
                    if number is not None:
                        col[s] = format_number(number)
                if name not in numeric:
                    numeric.append(name)
            elif action == TICK_NUMERIC:
                if name not in numeric:
                    numeric.append(name)
    return covs, numeric, removed


def describe_fixes(fixes: list[tuple[Issue, str]], removed: list[str]) -> str:
    """One line for the status bar."""
    done = [f for f in fixes if f[1] != NO_ACTION]
    text = f"Applied {len(done)} fix(es) to the covariates."
    if removed:
        text += " Removed: " + ", ".join(removed) + "."
    return text
