"""Deterministic Pandas analytics. All arithmetic happens here, never in the LLM."""
from __future__ import annotations

import calendar
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backend.analytics import chart_generator as charts

MONTHS: dict[str, int] = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
MONTHS["sept"] = 9
MONTH_RE = re.compile(r"\b(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

METRIC_GROUPS = [
    {"revenue", "sales", "turnover", "income", "amount"},
    {"profit", "margin"},
    {"cost", "expense", "spend"},
    {"units", "quantity", "qty", "volume", "orders"},
    {"price"},
]
DECLINE_RE = re.compile(r"\b(declin\w*|decreas\w*|drop\w*|fell|fall\w*|shrank|shrunk|loss|lost|down|decline[ds]?)\b")
GROWTH_RE = re.compile(r"\b(growth|grew|grow\w*|increas\w*|rose|rise|rising|gain\w*|up)\b")
CHANGE_RE = re.compile(r"\b(change[ds]?|changing|weakest|strongest)\b")
WEAK_RE = re.compile(r"\b(weakest|lowest|slowest|worst|smallest|poorest)\b")
TREND_RE = re.compile(r"\b(trend\w*|monthly|over time|each month|per month|by month|month by month|timeline)\b")
HIGH_RE = re.compile(r"\b(top|best|highest|most|largest|biggest|best-selling|top-selling|greatest|strongest)\b")
LOW_RE = re.compile(r"\b(worst|lowest|bottom|least|smallest|weakest|poorest)\b")
BREAKDOWN_RE = re.compile(r"\b(by|per|each|compare|comparison|versus|vs|breakdown|across|split)\b")
SHARE_RE = re.compile(r"\b(share|proportion|percentage of|mix|distribution)\b")
MULTIHOP_RE = re.compile(r"(best|top)[- ]selling|(best|top) (product|item)")
ID_LIKE = re.compile(r"(^|[_\s])(id|sku|code|zip|year|month|day|index)($|[_\s])")


class AnalyticsError(Exception):
    """Raised when a dataset cannot be loaded or understood."""


def fmt(value: float) -> str:
    return f"{value:,.2f}"


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None or (isinstance(value, float) and np.isnan(value)) else f"{value:+.1f}%"


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


def _is_text(s: pd.Series) -> bool:
    """True for object and pandas string dtypes (pandas 3 uses a dedicated string dtype)."""
    return s.dtype == object or pd.api.types.is_string_dtype(s)


def _to_numeric_if_possible(s: pd.Series) -> pd.Series:
    if not _is_text(s):
        return s
    cleaned = s.astype(str).str.replace(r"[,$€£₹%\s]", "", regex=True)
    conv = pd.to_numeric(cleaned, errors="coerce")
    non_null = int(s.notna().sum())
    if non_null and conv.notna().sum() / non_null >= 0.9:
        return conv
    return s


def load_csv(path: str | Path, source_name: str | None = None) -> "CSVAnalyzer":
    path = Path(path)
    df = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            try:
                df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            except pd.errors.ParserError:
                df = pd.read_csv(path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError as exc:
            raise AnalyticsError("The CSV file is empty.") from exc
        except Exception as exc:
            raise AnalyticsError(f"Could not parse CSV: {exc}") from exc
    if df is None:
        raise AnalyticsError("Could not decode the CSV file.")
    return CSVAnalyzer(df, source_name or path.name)


class CSVAnalyzer:
    """Understands one table: sums, averages, group-bys, rankings, trends, period changes."""

    def __init__(self, df: pd.DataFrame, source_name: str, label: str | None = None) -> None:
        if df is None or df.empty:
            raise AnalyticsError("The table has no rows.")
        self.source_name = source_name
        self.label = label or source_name
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
        if df.empty:
            raise AnalyticsError("The table has no usable data.")
        for c in df.columns:
            df[c] = _to_numeric_if_possible(df[c])
        self.df = df
        self.date_col: str | None = None
        self.month_only = False
        self._detect_dates()
        self.numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
            and not ID_LIKE.search(c.lower()) and not c.startswith("_")
        ]
        self.categorical_cols = [
            c for c in df.columns
            if c not in self.numeric_cols and c != self.date_col and not c.startswith("_")
            and not pd.api.types.is_datetime64_any_dtype(df[c])
            and df[c].nunique() <= 200 and (df[c].nunique() < len(df) or len(df) <= 20)
        ]
        self.last_issue = ""

    # ---------------------------------------------------------------- profile
    def _detect_dates(self) -> None:
        df = self.df
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.date_col = col
                df["_date"] = df[col]
                break
        if self.date_col is None:
            for col in df.columns:
                s = df[col]
                if not _is_text(s):
                    continue
                low = s.astype(str).str.strip().str.lower().str.rstrip(".")
                if low.isin(MONTHS.keys()).mean() >= 0.8:
                    nums = low.map(MONTHS)
                    built = pd.to_datetime(pd.DataFrame({"year": 2000, "month": nums.fillna(1).astype(int), "day": 1}))
                    df["_date"] = built.where(nums.notna())
                    self.date_col, self.month_only = col, True
                    break
                hinted = bool(re.search(r"date|month|period|time|day", col.lower()))
                looks_dated = s.astype(str).str.contains(r"\d{1,4}[-/.]\d{1,2}").mean() > 0.8
                if hinted or looks_dated:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        parsed = pd.to_datetime(s, errors="coerce")
                    if parsed.notna().mean() >= 0.8:
                        df["_date"] = parsed
                        self.date_col = col
                        break
        if self.date_col is None:
            lower = {c.lower(): c for c in df.columns}
            if "year" in lower and "month" in lower:
                y, m = df[lower["year"]], df[lower["month"]]
                if pd.api.types.is_numeric_dtype(y):
                    mnum = m if pd.api.types.is_numeric_dtype(m) else m.astype(str).str.strip().str.lower().map(MONTHS)
                    ok = mnum.notna() & y.notna()
                    if ok.mean() >= 0.8:
                        built = pd.to_datetime(pd.DataFrame({"year": y.fillna(2000).astype(int),
                                                             "month": mnum.fillna(1).astype(int), "day": 1}))
                        df["_date"] = built.where(ok)
                        self.date_col = f"{lower['year']}/{lower['month']}"
        if "_date" in df.columns:
            df["_period"] = df["_date"].dt.to_period("M")

    def describe(self) -> str:
        parts = [f"{c} ({self.df[c].dtype})" for c in self.df.columns if not c.startswith("_")]
        return f"{self.label}: {len(self.df)} rows; columns: " + ", ".join(parts)

    def summary(self) -> dict[str, Any]:
        return {"source": self.source_name, "label": self.label, "rows": int(len(self.df)),
                "columns": [c for c in self.df.columns if not c.startswith("_")],
                "date_column": self.date_col, "metrics": self.numeric_cols, "dimensions": self.categorical_cols}

    def plain_frame(self) -> pd.DataFrame:
        """DataFrame without internal helper columns (used by the SQL fallback)."""
        out = self.df[[c for c in self.df.columns if not c.startswith("_")]].copy()
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%Y-%m-%d")
        if "_date" in self.df.columns and self.date_col not in out.columns:
            out["date"] = self.df["_date"].dt.strftime("%Y-%m-%d")
        return out

    # ----------------------------------------------------------- understanding
    @staticmethod
    def _words(q: str) -> set[str]:
        words = set(re.findall(r"[a-z0-9]+", q.lower()))
        return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}

    def table_score(self, question: str) -> float:
        ql, words = question.lower(), self._words(question)
        score = 0.0
        for c in self.df.columns:
            if c.startswith("_"):
                continue
            if re.search(rf"\b{re.escape(c.lower().replace('_', ' '))}s?\b", ql):
                score += 2
        score += 2 * len(self._value_filters(question))
        for c in self.numeric_cols:
            for g in METRIC_GROUPS:
                if g & words and g & set(re.findall(r"[a-z]+", c.lower())):
                    score += 1.5
        stem = re.sub(r"\.[a-z]+$", "", self.source_name.lower())
        if stem and stem in ql:
            score += 3
        return score + min(len(self.df), 1_000_000) * 1e-9

    def _pick_metric(self, question: str) -> str | None:
        if not self.numeric_cols:
            return None
        words = self._words(question)
        best, best_score = None, 0
        for c in self.numeric_cols:
            toks = set(re.findall(r"[a-z]+", c.lower()))
            score = 3 if toks & words else 0
            for g in METRIC_GROUPS:
                if g & words and g & toks:
                    score = max(score, 2)
            if score > best_score:
                best, best_score = c, score
        if best:
            return best
        for hint in ("revenue", "sales", "amount", "profit", "total"):
            for c in self.numeric_cols:
                if hint in c.lower():
                    return c
        return self.numeric_cols[0]

    def _dimensions(self, question: str) -> list[str]:
        ql = question.lower()
        found = []
        for c in self.categorical_cols:
            m = re.search(rf"\b{re.escape(c.lower().replace('_', ' '))}(?:s|es)?\b", ql)
            if m:
                found.append((m.start(), c))
        return [c for _, c in sorted(found)]

    def _default_dimension(self) -> str | None:
        for hint in ("product", "category", "region", "segment", "customer"):
            for c in self.categorical_cols:
                if hint in c.lower():
                    return c
        return next((c for c in self.categorical_cols if 2 <= self.df[c].nunique() <= 50), None)

    def _value_filters(self, question: str) -> list[tuple[str, str]]:
        ql = question.lower()
        filters = []
        for c in self.categorical_cols:
            best = ""
            for v in self.df[c].dropna().astype(str).unique():
                s = v.strip()
                if len(s) >= 2 and len(s) > len(best) and re.search(rf"(?<!\w){re.escape(s.lower())}(?!\w)", ql):
                    best = s
            if best:
                filters.append((c, best))
        return filters

    @staticmethod
    def _time_mentions(question: str) -> tuple[list[int], list[int]]:
        ql = question.lower()
        months = [MONTHS[m] for m in MONTH_RE.findall(ql)]
        years = [int(y) for y in YEAR_RE.findall(ql)]
        return months, years

    def _filtered(self, question: str, apply_time: bool = True) -> tuple[pd.DataFrame, list[str]]:
        df, notes = self.df, []
        for col, val in self._value_filters(question):
            df = df[df[col].astype(str).str.strip() == val]
            notes.append(f"{col} = {val}")
        if apply_time and "_date" in df.columns:
            months, years = self._time_mentions(question)
            if months:
                df = df[df["_date"].dt.month.isin(months)]
                notes.append("month in " + ", ".join(calendar.month_name[m] for m in dict.fromkeys(months)))
            if years and not self.month_only:
                df = df[df["_date"].dt.year.isin(years)]
                notes.append("year in " + ", ".join(str(y) for y in dict.fromkeys(years)))
        return df, notes

    # ------------------------------------------------------------------ result
    def _result(self, intent: str, answer: str, calculation: str, data: Any, chart: Any,
                focus: dict[str, Any] | None, steps: list[str] | None = None) -> dict[str, Any]:
        return {"ok": True, "intent": intent, "answer": answer, "calculation": calculation,
                "data": data if isinstance(data, list) else records(data), "chart": chart, "focus": focus,
                "steps": steps or [], "source": self.source_name, "table": self.label}

    # ---------------------------------------------------------------- dispatch
    def analyze(self, question: str) -> dict[str, Any] | None:
        self.last_issue = ""
        ql = question.lower()
        metric = self._pick_metric(question)
        dims = self._dimensions(question)
        changeish = bool(DECLINE_RE.search(ql) or GROWTH_RE.search(ql) or CHANGE_RE.search(ql))
        if changeish and "_period" not in self.df.columns:
            self.last_issue = "No date column found, so period changes can't be calculated."
            changeish = False
        if metric is None and not re.search(r"\b(count|how many|number of)\b", ql):
            self.last_issue = "No numeric column found in this table."
            return None
        try:
            if changeish and metric:
                if MULTIHOP_RE.search(ql) and len(dims) >= 2:
                    return self._multihop(question, metric, dims)
                return self._change(question, metric, dims[0] if dims else None)
            if TREND_RE.search(ql) and metric and "_period" in self.df.columns:
                return self._trend(question, metric, dims[0] if dims else None)
            if dims:
                return self._group(question, metric, dims[0])
            return self._scalar(question, metric)
        except (KeyError, ValueError, ZeroDivisionError) as exc:
            self.last_issue = f"Could not compute this with the rule-based analyzer: {exc}"
            return None

    # ------------------------------------------------------------------ scalar
    def _scalar(self, question: str, metric: str | None) -> dict[str, Any] | None:
        ql = question.lower()
        df, notes = self._filtered(question)
        if df.empty:
            self.last_issue = "No rows match the filters in the question."
            return None
        where = (" WHERE " + " AND ".join(notes)) if notes else ""
        if re.search(r"\b(count|how many|number of)\b", ql) or metric is None:
            value, agg, label = len(df), "COUNT", "Number of rows"
            calc = f"COUNT(*){where} = {value:,}"
            text = f"{label}{(' (' + '; '.join(notes) + ')') if notes else ''}: {value:,}."
        else:
            if re.search(r"\b(average|avg|mean)\b", ql):
                value, agg, label = float(df[metric].mean()), "AVG", "Average"
            elif re.search(r"\b(maximum|max|highest|largest|biggest)\b", ql):
                value, agg, label = float(df[metric].max()), "MAX", "Maximum"
            elif re.search(r"\b(minimum|min|lowest|smallest)\b", ql):
                value, agg, label = float(df[metric].min()), "MIN", "Minimum"
            else:
                value, agg, label = float(df[metric].sum()), "SUM", "Total"
            calc = f"{agg}({metric}){where} over {len(df):,} rows = {fmt(value)}"
            text = f"{label} {metric}{(' (' + '; '.join(notes) + ')') if notes else ''}: {fmt(value)}."
        data = [{"measure": agg, "value": value, "rows": int(len(df))}]
        return self._result("scalar", text, calc, data, None, None)

    # ------------------------------------------------------------------- group
    def _group(self, question: str, metric: str | None, dim: str) -> dict[str, Any] | None:
        ql = question.lower()
        df, notes = self._filtered(question)
        df = df[~df[dim].isna()]
        if df.empty:
            self.last_issue = "No rows match the filters in the question."
            return None
        count_mode = bool(re.search(r"\b(count|how many|number of)\b", ql)) or metric is None
        if count_mode:
            g = df.groupby(dim).size().rename("value")
            agg = "COUNT"
        elif re.search(r"\b(average|avg|mean)\b", ql):
            g, agg = df.groupby(dim)[metric].mean().rename("value"), "AVG"
        else:
            g, agg = df.groupby(dim)[metric].sum().rename("value"), "SUM"
        ascending = bool(LOW_RE.search(ql)) and not HIGH_RE.search(ql)
        g = g.sort_values(ascending=ascending).reset_index()
        n_match = re.search(r"\b(?:top|bottom)\s+(\d+)\b", ql)
        limit = int(n_match.group(1)) if n_match else (10 if BREAKDOWN_RE.search(ql) else 5)
        shown = g.head(limit)
        total = float(g["value"].sum())
        shown = shown.assign(share_pct=(shown["value"] / total * 100).round(1) if total else np.nan)
        measure = "rows" if count_mode else metric
        lines = [f"{i}. {row[dim]}: {fmt(row['value'])}" for i, row in enumerate(shown.to_dict('records'), 1)]
        top = shown.iloc[0]
        adjective = "lowest" if ascending else "highest"
        notes_txt = f" ({'; '.join(notes)})" if notes else ""
        text = (f"{top[dim]} has the {adjective} {agg.lower() if agg != 'SUM' else 'total'} {measure}"
                f"{notes_txt}: {fmt(top['value'])}.\n" + "\n".join(lines))
        where = (" WHERE " + " AND ".join(notes)) if notes else ""
        calc = f"{agg}({metric if not count_mode else '*'}){where} GROUP BY {dim} ORDER BY value {'ASC' if ascending else 'DESC'}"
        chart = None
        if charts.should_chart("rank", len(shown)):
            if SHARE_RE.search(ql):
                chart = charts.pie_chart(shown[dim].astype(str), shown["value"], f"Share of {measure} by {dim}")
            else:
                chart = charts.bar_chart(shown[dim].astype(str), shown["value"], f"{agg.title()} {measure} by {dim}",
                                         dim, measure)
        focus = {"entities": [str(top[dim])], "dimension": dim, "metric": metric,
                 "direction": "low" if ascending else "high", "periods": []}
        return self._result("rank", text, calc, shown, chart, focus)

    # ------------------------------------------------------------------- trend
    def _trend(self, question: str, metric: str, dim: str | None) -> dict[str, Any] | None:
        df, notes = self._filtered(question)
        df = df.dropna(subset=["_period"])
        if df.empty:
            self.last_issue = "No dated rows match the question."
            return None
        by_period = df.groupby("_period")[metric].sum().sort_index()
        labels = [str(p) if not self.month_only else calendar.month_name[p.month] for p in by_period.index]
        first, last = float(by_period.iloc[0]), float(by_period.iloc[-1])
        pct = (last - first) / first * 100 if first else None
        peak_i, low_i = int(by_period.values.argmax()), int(by_period.values.argmin())
        text = (f"Monthly {metric}{(' (' + '; '.join(notes) + ')') if notes else ''}: {fmt(first)} in {labels[0]} "
                f"to {fmt(last)} in {labels[-1]} ({fmt_pct(pct)}). Peak: {labels[peak_i]} ({fmt(by_period.iloc[peak_i])}); "
                f"lowest: {labels[low_i]} ({fmt(by_period.iloc[low_i])}).")
        series = {metric: (labels, by_period.values)}
        data_df = pd.DataFrame({"period": labels, metric: by_period.values})
        if dim:
            top_vals = df.groupby(dim)[metric].sum().nlargest(6).index
            piv = df[df[dim].isin(top_vals)].pivot_table(index="_period", columns=dim, values=metric,
                                                        aggfunc="sum").sort_index()
            series = {str(c): ([str(p) for p in piv.index], piv[c].fillna(0).values) for c in piv.columns}
            data_df = piv.reset_index().assign(_period=lambda d: d["_period"].astype(str)).rename(columns={"_period": "period"})
        chart = charts.line_chart(series, f"{metric} by month", "Month", metric) if len(by_period) >= 2 else None
        where = (" WHERE " + " AND ".join(notes)) if notes else ""
        calc = f"SUM({metric}){where} GROUP BY month" + (f", {dim}" if dim else "")
        return self._result("trend", text, calc, data_df, chart, {"entities": [], "dimension": dim, "metric": metric,
                                                                  "direction": None, "periods": [labels[0], labels[-1]]})

    # ------------------------------------------------------------------ change
    def _choose_periods(self, question: str, growth_mode: bool, df: pd.DataFrame) -> tuple[pd.Period, pd.Period, str] | None:
        periods = sorted(df["_period"].dropna().unique())
        if len(periods) < 2:
            return None
        months, years = self._time_mentions(question)

        def resolve(month: int) -> pd.Period | None:
            cands = [p for p in periods if p.month == month and (not years or p.year in years or self.month_only)]
            return cands[-1] if cands else None

        if len(set(months)) >= 2:
            a, b = resolve(months[0]), resolve(months[1])
            if a and b and a != b:
                return a, b, f"comparing {a} to {b}"
        if months:
            to = resolve(months[0])
            if to is not None:
                prev = to - 1
                if prev in periods:
                    return prev, to, f"{to} vs the previous month {prev}"
                earlier = [p for p in periods if p < to]
                if earlier:
                    return earlier[-1], to, f"{to} vs the previous available period {earlier[-1]}"
        if growth_mode:
            return periods[0], periods[-1], f"first period {periods[0]} to last period {periods[-1]}"
        return periods[-2], periods[-1], f"latest two periods {periods[-2]} → {periods[-1]}"

    def _change_table(self, df: pd.DataFrame, metric: str, dim: str, p_from: pd.Period, p_to: pd.Period) -> pd.DataFrame:
        sub = df[df["_period"].isin([p_from, p_to])]
        piv = sub.pivot_table(index=dim, columns="_period", values=metric, aggfunc="sum", fill_value=0.0)
        for p in (p_from, p_to):
            if p not in piv.columns:
                piv[p] = 0.0
        out = pd.DataFrame({dim: piv.index, str(p_from): piv[p_from].values, str(p_to): piv[p_to].values})
        out["change"] = out[str(p_to)] - out[str(p_from)]
        out["pct_change"] = np.where(out[str(p_from)] != 0, out["change"] / out[str(p_from)].replace(0, np.nan) * 100, np.nan)
        return out

    def _direction(self, ql: str) -> tuple[str, str]:
        decline, growth, weak = bool(DECLINE_RE.search(ql)), bool(GROWTH_RE.search(ql)), bool(WEAK_RE.search(ql))
        if growth and weak:
            return "weakest_growth", "weakest growth"
        if decline and not growth:
            return "decline", "largest decline"
        if growth:
            return "growth", "largest growth"
        return "abs", "largest change"

    def _pick_row(self, table: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.Series]:
        if mode == "weakest_growth":
            t = table.sort_values("pct_change", ascending=True, na_position="last")
        elif mode == "decline":
            t = table.sort_values("change", ascending=True)
        elif mode == "growth":
            t = table.sort_values("change", ascending=False)
        else:
            t = table.reindex(table["change"].abs().sort_values(ascending=False).index)
        return t.reset_index(drop=True), t.iloc[0]

    def _change(self, question: str, metric: str, dim: str | None) -> dict[str, Any] | None:
        ql = question.lower()
        df, notes = self._filtered(question, apply_time=False)
        df = df.dropna(subset=["_period"])
        mode, label = self._direction(ql)
        chosen = self._choose_periods(question, mode in {"growth", "weakest_growth"}, df)
        if chosen is None:
            self.last_issue = "Need at least two distinct months of data to calculate a change."
            return None
        p_from, p_to, how = chosen
        a, b = str(p_from), str(p_to)
        filt = f" ({'; '.join(notes)})" if notes else ""

        if dim:
            table = self._change_table(df, metric, dim, p_from, p_to)
            ranked, row = self._pick_row(table, mode)
            pct = None if pd.isna(row["pct_change"]) else float(row["pct_change"])
            none_declined = mode == "decline" and row["change"] >= 0
            head = (f"No {dim} declined ({how}); the smallest increase was {row[dim]}." if none_declined
                    else f"{row[dim]} had the {label} in {metric}{filt} ({how}).")
            text = (f"{head} {fmt(row[a])} ({a}) → {fmt(row[b])} ({b}), a change of "
                    f"{fmt(row['change'])} ({fmt_pct(pct)}).")
            calc = (f"SUM({metric}) GROUP BY {dim} for {a} and {b}; change = {b} − {a}; "
                    f"% change = change ÷ {a} × 100; ranked by "
                    f"{'% change' if mode == 'weakest_growth' else 'absolute change'}")
            top = ranked.head(10)
            chart = charts.bar_chart(top[dim].astype(str), top["change"].round(2),
                                     f"Change in {metric}: {a} → {b}", dim, "Change", signed_colors=True) \
                if charts.should_chart("change", len(top)) else None
            focus = {"entities": [str(row[dim])], "dimension": dim, "metric": metric,
                     "direction": "decline" if row["change"] < 0 else "growth", "periods": [a, b]}
            return self._result("change", text, calc, ranked.round(2), chart, focus)

        total_from = float(df[df["_period"] == p_from][metric].sum())
        total_to = float(df[df["_period"] == p_to][metric].sum())
        delta = total_to - total_from
        pct = delta / total_from * 100 if total_from else None
        verb = "declined" if delta < 0 else "grew" if delta > 0 else "was flat"
        text = (f"Total {metric}{filt} {verb} from {fmt(total_from)} ({a}) to {fmt(total_to)} ({b}): "
                f"{fmt(delta)} ({fmt_pct(pct)}) — {how}.")
        calc = f"SUM({metric}) for {a} and {b}; change = {b} − {a}; % change = change ÷ {a} × 100"
        data: Any = [{"period": a, metric: total_from}, {"period": b, metric: total_to},
                     {"period": "change", metric: delta, "pct_change": pct}]
        entities = [v for _, v in self._value_filters(question)]
        focus = {"entities": entities, "dimension": None, "metric": metric,
                 "direction": "decline" if delta < 0 else "growth", "periods": [a, b]}
        chart = None
        default_dim = None if entities else self._default_dimension()
        if default_dim:
            table = self._change_table(df, metric, default_dim, p_from, p_to)
            ranked, row = self._pick_row(table, "decline" if delta < 0 else "growth")
            text += (f" Biggest contributor ({default_dim}): {row[default_dim]} "
                     f"({fmt(row['change'])}, {fmt_pct(None if pd.isna(row['pct_change']) else float(row['pct_change']))}).")
            top = ranked.head(10)
            chart = charts.bar_chart(top[default_dim].astype(str), top["change"].round(2),
                                     f"Change in {metric} by {default_dim}: {a} → {b}", default_dim, "Change",
                                     signed_colors=True)
            focus.update({"entities": [str(row[default_dim])], "dimension": default_dim})
            data = data + records(ranked.round(2))
            return self._result("change", text, calc + f"; contributors via GROUP BY {default_dim}", data, chart, focus)
        return self._result("change", text, calc, data, chart, focus)

    # --------------------------------------------------------------- multi-hop
    def _multihop(self, question: str, metric: str, dims: list[str]) -> dict[str, Any] | None:
        ql = question.lower()
        df, _ = self._filtered(question, apply_time=False)
        df = df.dropna(subset=["_period"])
        dim1, dim2 = dims[0], dims[1]
        mode, label = self._direction(ql)
        chosen = self._choose_periods(question, mode in {"growth", "weakest_growth"}, df)
        if chosen is None:
            self.last_issue = "Need at least two distinct months of data for a multi-step analysis."
            return None
        p_from, p_to, how = chosen
        a, b = str(p_from), str(p_to)
        table = self._change_table(df, metric, dim1, p_from, p_to)
        ranked, row = self._pick_row(table, mode)
        entity1 = str(row[dim1])
        pct1 = None if pd.isna(row["pct_change"]) else float(row["pct_change"])
        step1 = (f"Step 1: {entity1} is the {dim1} with the {label} in {metric} ({how}): "
                 f"{fmt(row[a])} → {fmt(row[b])} ({fmt(row['change'])}, {fmt_pct(pct1)}).")
        sub = df[df[dim1].astype(str) == entity1]
        best = sub.groupby(dim2)[metric].sum().sort_values(ascending=False)
        entity2 = str(best.index[0])
        step2 = (f"Step 2: within {entity1}, the best-selling {dim2} is {entity2} "
                 f"with total {metric} of {fmt(best.iloc[0])} across all periods.")
        chart = charts.bar_chart(best.head(10).index.astype(str), best.head(10).values.round(2),
                                 f"{metric} by {dim2} in {entity1}", dim2, metric)
        calc = (f"1) SUM({metric}) GROUP BY {dim1} for {a} and {b}, ranked; 2) filter {dim1} = {entity1}, "
                f"SUM({metric}) GROUP BY {dim2}, take max")
        data = [{dim1: entity1, "step": 1, a: float(row[a]), b: float(row[b]), "change": float(row["change"]),
                 "pct_change": pct1}] + [{dim2: str(k), "step": 2, "total": float(v)} for k, v in best.head(10).items()]
        focus = {"entities": [entity1, entity2], "dimension": dim1, "metric": metric,
                 "direction": "decline" if row["change"] < 0 else "growth", "periods": [a, b]}
        return self._result("multihop", step1 + "\n" + step2, calc, data, chart, focus, steps=[step1, step2])
