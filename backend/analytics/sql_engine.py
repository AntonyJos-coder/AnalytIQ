"""Read-only SQL fallback: the LLM may *write* a SELECT, but SQLite does the calculation."""
from __future__ import annotations

import re
import sqlite3
from typing import Any

import pandas as pd

from backend.analytics import chart_generator as charts
from backend.analytics.csv_analyzer import CSVAnalyzer, fmt, records
from backend.rag.generator import LMStudioClient

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|replace|vacuum|truncate|load_extension)\b", re.I)
MAX_ROWS = 200


class UnsafeSQLError(ValueError):
    pass


class SQLEngine:
    def __init__(self, analyzer: CSVAnalyzer) -> None:
        frame = analyzer.plain_frame()
        frame.columns = [re.sub(r"\W+", "_", str(c)).strip("_") or f"col{i}" for i, c in enumerate(frame.columns)]
        self.source = analyzer.source_name
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        frame.to_sql("data", self.conn, index=False)
        sample = frame.head(3).to_dict("records")
        self.schema = "Table data(" + ", ".join(f"{c} {t}" for c, t in
                                                  zip(frame.columns, [str(d) for d in frame.dtypes])) + ")"
        self.sample = sample

    @staticmethod
    def validate(sql: str) -> str:
        sql = re.sub(r"^```(?:sql)?|```$", "", sql.strip(), flags=re.I | re.M).strip().rstrip(";").strip()
        if not re.match(r"^(select|with)\b", sql, re.I) or ";" in sql or FORBIDDEN.search(sql):
            raise UnsafeSQLError("Only a single read-only SELECT statement is allowed.")
        return sql

    def run(self, sql: str) -> pd.DataFrame:
        sql = self.validate(sql)
        return pd.read_sql_query(f"SELECT * FROM ({sql}) LIMIT {MAX_ROWS}", self.conn)

    def answer(self, question: str, llm: LMStudioClient) -> dict[str, Any] | None:
        prompt = (
            "Write ONE SQLite SELECT query that answers the question. Use only the table and columns given. "
            "Return only the SQL, no explanation.\n"
            f"{self.schema}\nSample rows: {self.sample}\nQuestion: {question}"
        )
        sql = llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=250)
        df = self.run(sql)
        if df.empty:
            return None
        sql_clean = self.validate(sql)
        if df.shape == (1, 1):
            v = df.iat[0, 0]
            text = f"Result: {fmt(float(v)) if isinstance(v, (int, float)) else v}."
        elif len(df) <= 10:
            text = "Result:\n" + "\n".join(", ".join(f"{k}: {v}" for k, v in row.items()) for row in df.to_dict("records"))
        else:
            text = f"The query returned {len(df)} rows (showing the data table)."
        chart = None
        if df.shape[1] == 2 and 2 <= len(df) <= 30 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]) \
                and not pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
            chart = charts.bar_chart(df.iloc[:, 0].astype(str), df.iloc[:, 1], question[:60])
        return {"ok": True, "intent": "sql", "answer": text, "calculation": f"SQL executed by SQLite: {sql_clean}",
                "data": records(df), "chart": chart, "focus": None, "steps": [], "source": self.source,
                "table": self.source}
