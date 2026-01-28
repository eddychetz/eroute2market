# EROUTE2MARKET
# ER2M DATA SCIENCE TEAM
# ***
# Parsers

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import BaseOutputParser

import re
import pandas as pd
from datetime import datetime

# Python Parser for output standardization
class PythonOutputParser(BaseOutputParser):
    def parse(self, text: str):
        def extract_python_code(text):
            python_code_match = re.search(r'```python(.*?)```', text, re.DOTALL)
            if python_code_match:
                python_code = python_code_match.group(1).strip()
                return python_code
            else:
                python_code_match = re.search(r"python(.*?)'", text, re.DOTALL)
                if python_code_match:
                    python_code = python_code_match.group(1).strip()
                    return python_code
                else:
                    return None
        python_code = extract_python_code(text)
        if python_code is not None:
            return python_code
        else:
            # Assume ```sql wasn't used
            return text

# Date Parser for output standardization
class DateOutputParser(BaseOutputParser):
    def parse(self, text: str):

        # Define a function to parse a single date
        def parse_single_date(token):

            POSSIBLE_FMTS = [
                "%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y", "%d-%m-%Y", "%d-%m-%y",
                "%d %b %Y", "%d %b '%y", "%d %B %Y", "%d %B '%y"
            ]

            token = str(token).strip().replace("\u2009", " ")
            for fmt in POSSIBLE_FMTS:
                try:
                    dt = datetime.strptime(token, fmt)
                    if dt.year < 100:  # normalize 2-digit years to 2000s
                        dt = dt.replace(year=2000 + dt.year)
                    return pd.Timestamp(dt)
                except Exception:
                    pass
            return pd.to_datetime(token, dayfirst=True, errors="coerce")

        # Define a function to split a range of date strings
        def parser_split_range(cell):
            """
            Return (start, finish) parsed from a range string or single date.
            """
            DASH_PATTERN = re.compile(r"\s*[\-–—]\s*")  # hyphen, en dash, em dash
            DATE_TOKEN = re.compile(
                r"(\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}\s*[A-Za-z]{3,9}\s*'\d{2,4}|\d{1,2}\s*[A-Za-z]{3,9}\s*\d{2,4})"
                                    )
            if pd.isna(cell):
                return (pd.NaT, pd.NaT)
            s = str(cell).strip()
            if not s:
                return (pd.NaT, pd.NaT)

            # 1) Try splitting by any dash
            parts = DASH_PATTERN.split(s)
            if len(parts) == 2:
                return (self.parse_single_date(parts[0]), self.parse_single_date(parts[1]))

            # 2) Otherwise, take the first 1–2 recognizable date tokens
            tokens = DATE_TOKEN.findall(s)
            if len(tokens) >= 2:
                return (self.parse_single_date(tokens[0]), self.parse_single_date(tokens[1]))
            if len(tokens) == 1:
                return (self.parse_single_date(tokens[0]), pd.NaT)
            return (pd.NaT, pd.NaT)


# SQL Parser for output standardization
class SQLOutputParser(BaseOutputParser):
    def parse(self, text: str):
        def extract_sql_code(text):
            sql_code_match = re.search(r'```sql(.*?)```', text, re.DOTALL)
            sql_code_match_2 = re.search(r"SQLQuery:\s*(.*)", text)
            if sql_code_match:
                sql_code = sql_code_match.group(1).strip()
                return sql_code
            if sql_code_match_2:
                sql_code = sql_code_match_2.group(1).strip()
                return sql_code
            else:
                sql_code_match = re.search(r"sql(.*?)'", text, re.DOTALL)
                if sql_code_match:
                    sql_code = sql_code_match.group(1).strip()
                    return sql_code
                else:
                    return None
        sql_code = extract_sql_code(text)
        if sql_code is not None:
            return sql_code
        else:
            # Assume ```sql wasn't used
            return text