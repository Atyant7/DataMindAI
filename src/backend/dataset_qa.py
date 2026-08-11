"""
Deterministic dataset question-answering engine.

The DatasetQA engine performs calculations directly against
the uploaded pandas DataFrame.

The engine intentionally does NOT use an LLM.

Its purpose is to provide verified evidence that can later
be consumed by the LangChain + LangGraph + local LLM layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ============================================================
# RESULT OBJECT
# ============================================================


@dataclass
class DatasetQAResult:
    """
    Standardized result returned by DatasetQA.
    """

    success: bool

    question: str

    operation: str

    answer: Any = None

    evidence: dict[str, Any] = field(
        default_factory=dict
    )

    dataframe: pd.DataFrame | None = None

    error: str | None = None

    warnings: list[str] = field(
        default_factory=list
    )

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the result to a serializable dictionary.
        """

        result = {
            "success": self.success,
            "question": self.question,
            "operation": self.operation,
            "answer": self.answer,
            "evidence": self.evidence,
            "error": self.error,
            "warnings": self.warnings,
        }

        if self.dataframe is not None:

            result["dataframe"] = (
                self.dataframe.to_dict(
                    orient="records"
                )
            )

        else:

            result["dataframe"] = None

        return result


# ============================================================
# DATASET QA ENGINE
# ============================================================


class DatasetQA:
    """
    Deterministic question-answering engine for tabular data.

    Supported operations include:

    - dataset overview
    - column listing
    - missing-value analysis
    - duplicate analysis
    - numerical statistics
    - categorical value counts
    - unique-value counts
    - correlation
    - group-by aggregation
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
    ) -> None:

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "dataframe must be a pandas DataFrame."
            )

        if dataframe.empty:

            raise ValueError(
                "The dataset cannot be empty."
            )

        self.df = dataframe.copy()

        self.columns = (
            self.df.columns.tolist()
        )

    # ========================================================
    # PUBLIC API
    # ========================================================

    def answer(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Analyze a natural-language dataset question.

        The current version uses deterministic intent detection.
        Later LangGraph nodes can convert LLM output into the
        same structured operations.
        """

        if not question or not question.strip():

            return DatasetQAResult(
                success=False,
                question=question,
                operation="unknown",
                error="Question cannot be empty.",
            )

        normalized = (
            question
            .strip()
            .lower()
        )

        try:

            operation = (
                self._detect_operation(
                    normalized
                )
            )

            if operation == "overview":

                return self._overview(
                    question
                )

            if operation == "columns":

                return self._columns(
                    question
                )

            if operation == "missing_values":

                return self._missing_values(
                    question
                )

            if operation == "duplicates":

                return self._duplicates(
                    question
                )

            if operation == "statistics":

                return self._statistics(
                    question
                )

            if operation == "unique_values":

                return self._unique_values(
                    question
                )

            if operation == "value_counts":

                return self._value_counts(
                    question
                )

            if operation == "correlation":

                return self._correlation(
                    question
                )

            if operation == "groupby":

                return self._groupby(
                    question
                )

            return DatasetQAResult(
                success=False,
                question=question,
                operation="unknown",
                error=(
                    "I could not determine a supported "
                    "data-analysis operation from the question."
                ),
            )

        except Exception as exc:

            return DatasetQAResult(
                success=False,
                question=question,
                operation="error",
                error=str(exc),
            )

    # ========================================================
    # INTENT DETECTION
    # ========================================================

    def _detect_operation(
        self,
        question: str,
    ) -> str:
        """
        Detect the dataset operation requested by the user.
        """

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "missing value",
                "missing values",
                "null value",
                "null values",
                "nulls",
                "nans",
                "na values",
                "empty values",
            )
        ):

            return "missing_values"

        # ----------------------------------------------------
        # Duplicate rows
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "duplicate",
                "duplicates",
                "duplicate rows",
                "repeated rows",
            )
        ):

            return "duplicates"

        # ----------------------------------------------------
        # Correlation
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "correlation",
                "correlated",
                "relationship between",
                "relationship of",
            )
        ):

            return "correlation"

        # ----------------------------------------------------
        # Unique values
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "unique values",
                "unique value",
                "distinct values",
                "distinct value",
                "how many unique",
                "number of unique",
            )
        ):

            return "unique_values"

        # ----------------------------------------------------
        # Value counts / frequency
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "value counts",
                "frequency",
                "frequencies",
                "most common",
                "most frequent",
                "distribution of",
                "how many are",
            )
        ):

            return "value_counts"

        # ----------------------------------------------------
        # Dataset columns
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "what columns",
                "which columns",
                "list columns",
                "column names",
                "columns in",
                "features in",
            )
        ):

            return "columns"

        # ----------------------------------------------------
        # Dataset overview
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "how many rows",
                "how many columns",
                "dataset size",
                "dataset shape",
                "shape of",
                "overview",
                "describe the dataset",
                "about the dataset",
                "information about the dataset",
            )
        ):

            return "overview"

        # ----------------------------------------------------
        # Group-by
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "group by",
                "grouped by",
                "per category",
                "by category",
                "for each",
                "for every",
            )
        ):

            return "groupby"

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        if any(
            phrase in question
            for phrase in (
                "average",
                "mean",
                "median",
                "minimum",
                "minimum value",
                "maximum",
                "maximum value",
                "lowest",
                "highest",
                "sum",
                "total",
                "standard deviation",
                "std",
                "variance",
            )
        ):

            return "statistics"

        return "unknown"

    # ========================================================
    # COLUMN MATCHING
    # ========================================================

    def _find_column(
        self,
        question: str,
    ) -> str | None:
        """
        Find the most relevant dataset column mentioned
        in the user's question.

        Matching is token-based rather than substring-based,
        so a column such as ``age`` is not incorrectly matched
        inside a word such as ``average``.
        """

        normalized_question = self._normalize_text(
            question
        )

        question_tokens = normalized_question.split()

        normalized_columns = [
            (
                column,
                self._normalize_text(str(column)),
            )
            for column in self.columns
        ]

        def singular_forms(word: str) -> set[str]:
            """
            Return simple singular/plural variants.
            """

            forms = {word}

            if word.endswith("ies") and len(word) > 3:
                forms.add(word[:-3] + "y")

            elif word.endswith("y") and len(word) > 1:
                forms.add(word[:-1] + "ies")

            elif word.endswith("s") and len(word) > 1:
                forms.add(word[:-1])

            else:
                forms.add(word + "s")

            return forms

        # ----------------------------------------------------
        # Exact single-word column match
        # ----------------------------------------------------

        for column, normalized_column in normalized_columns:

            if (
                " " not in normalized_column
                and normalized_column in question_tokens
            ):
                return str(column)

        # ----------------------------------------------------
        # Exact multi-word column phrase match
        # ----------------------------------------------------

        for column, normalized_column in normalized_columns:

            column_tokens = normalized_column.split()

            if len(column_tokens) <= 1:
                continue

            column_length = len(column_tokens)

            for index in range(
                len(question_tokens) - column_length + 1
            ):

                phrase = " ".join(
                    question_tokens[
                        index:index + column_length
                    ]
                )

                if phrase == normalized_column:
                    return str(column)

        # ----------------------------------------------------
        # Singular/plural matching
        # ----------------------------------------------------

        for column, normalized_column in normalized_columns:

            if " " in normalized_column:
                continue

            column_forms = singular_forms(
                normalized_column
            )

            if any(
                token in column_forms
                for token in question_tokens
            ):
                return str(column)

        # ----------------------------------------------------
        # Token-overlap fallback
        # ----------------------------------------------------

        best_column = None
        best_score = 0

        question_token_set = set(
            question_tokens
        )

        for column, normalized_column in normalized_columns:

            column_tokens = set(
                normalized_column.split()
            )

            expanded_column_tokens = set()

            for token in column_tokens:
                expanded_column_tokens.update(
                    singular_forms(token)
                )

            overlap = (
                question_token_set
                & expanded_column_tokens
            )

            score = len(overlap)

            if score > best_score:
                best_score = score
                best_column = column

        if best_score > 0:
            return str(best_column)

        return None

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text for column matching.
        """

        import re

        normalized = (
            str(text)
            .lower()
            .replace("_", " ")
            .replace("-", " ")
            .replace("/", " ")
        )

        # Remove punctuation while preserving spaces so
        # "salary?" becomes "salary" and "cities," becomes
        # "cities".
        normalized = re.sub(
            r"[^\w\s]",
            " ",
            normalized,
        )

        return " ".join(
            normalized.split()
        )

    def _require_column(
        self,
        question: str,
    ) -> str:
        """
        Find a column or raise a meaningful error.
        """

        column = self._find_column(
            question
        )

        if column is None:

            raise ValueError(
                "I could not identify a dataset column "
                "from your question."
            )

        return column

    # ========================================================
    # OVERVIEW
    # ========================================================

    def _overview(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Return basic dataset information.
        """

        rows, columns = (
            self.df.shape
        )

        missing = int(
            self.df.isna()
            .sum()
            .sum()
        )

        duplicates = int(
            self.df.duplicated()
            .sum()
        )

        answer = (
            f"The dataset contains {rows:,} rows "
            f"and {columns:,} columns. "
            f"It has {missing:,} missing values "
            f"and {duplicates:,} duplicate rows."
        )

        evidence = {
            "rows": rows,
            "columns": columns,
            "missing_values": missing,
            "duplicate_rows": duplicates,
            "column_names": [
                str(column)
                for column in self.columns
            ],
        }

        return DatasetQAResult(
            success=True,
            question=question,
            operation="overview",
            answer=answer,
            evidence=evidence,
        )

    # ========================================================
    # COLUMNS
    # ========================================================

    def _columns(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Return dataset column names.
        """

        column_names = [
            str(column)
            for column in self.columns
        ]

        answer = (
            "The dataset contains "
            f"{len(column_names)} columns: "
            + ", ".join(
                column_names
            )
            + "."
        )

        return DatasetQAResult(
            success=True,
            question=question,
            operation="columns",
            answer=answer,
            evidence={
                "columns": column_names,
                "count": len(
                    column_names
                ),
            },
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    def _missing_values(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Analyze missing values.
        """

        missing_counts = (
            self.df.isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        missing_counts = (
            missing_counts[
                missing_counts > 0
            ]
        )

        total_missing = int(
            missing_counts.sum()
        )

        if missing_counts.empty:

            answer = (
                "The dataset does not contain "
                "any missing values."
            )

        else:

            top_missing = (
                missing_counts
                .head(5)
                .to_dict()
            )

            details = ", ".join(
                f"{column}: {int(count)}"
                for column, count
                in top_missing.items()
            )

            answer = (
                f"The dataset contains "
                f"{total_missing:,} missing values. "
                f"The columns with the most missing "
                f"values are {details}."
            )

        evidence = {
            "total_missing_values": total_missing,
            "columns_with_missing_values": {
                str(column): int(count)
                for column, count
                in missing_counts.items()
            },
        }

        return DatasetQAResult(
            success=True,
            question=question,
            operation="missing_values",
            answer=answer,
            evidence=evidence,
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    def _duplicates(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Analyze duplicate rows.
        """

        duplicate_count = int(
            self.df.duplicated()
            .sum()
        )

        if duplicate_count == 0:

            answer = (
                "The dataset contains "
                "no duplicate rows."
            )

        else:

            percentage = (
                duplicate_count
                / len(self.df)
                * 100
            )

            answer = (
                f"The dataset contains "
                f"{duplicate_count:,} duplicate rows "
                f"({percentage:.2f}% of all rows)."
            )

        return DatasetQAResult(
            success=True,
            question=question,
            operation="duplicates",
            answer=answer,
            evidence={
                "duplicate_rows": duplicate_count,
                "duplicate_percentage": (
                    duplicate_count
                    / len(self.df)
                    * 100
                ),
            },
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def _statistics(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Calculate a requested statistic for a column.
        """

        column = self._require_column(
            question
        )

        series = self.df[
            column
        ]

        if not pd.api.types.is_numeric_dtype(
            series
        ):

            raise ValueError(
                f"Column '{column}' is not numeric, "
                "so numerical statistics cannot be calculated."
            )

        question_lower = (
            question.lower()
        )

        if "median" in question_lower:

            operation = "median"

            value = float(
                series.median()
            )

        elif (
            "minimum" in question_lower
            or "lowest" in question_lower
            or "smallest" in question_lower
        ):

            operation = "minimum"

            value = float(
                series.min()
            )

        elif (
            "maximum" in question_lower
            or "highest" in question_lower
            or "largest" in question_lower
        ):

            operation = "maximum"

            value = float(
                series.max()
            )

        elif (
            "sum" in question_lower
            or "total" in question_lower
        ):

            operation = "sum"

            value = float(
                series.sum()
            )

        elif (
            "standard deviation"
            in question_lower
            or "std"
            in question_lower
        ):

            operation = "standard_deviation"

            value = float(
                series.std()
            )

        elif "variance" in question_lower:

            operation = "variance"

            value = float(
                series.var()
            )

        else:

            operation = "mean"

            value = float(
                series.mean()
            )

        answer = (
            f"The {operation.replace('_', ' ')} "
            f"of '{column}' is {value:,.4f}."
        )

        return DatasetQAResult(
            success=True,
            question=question,
            operation=operation,
            answer=answer,
            evidence={
                "column": column,
                "value": value,
                "count": int(
                    series.count()
                ),
            },
        )

    # ========================================================
    # UNIQUE VALUES
    # ========================================================

    def _unique_values(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Return the number of unique values in a column.
        """

        column = self._require_column(
            question
        )

        count = int(
            self.df[
                column
            ].nunique(
                dropna=True
            )
        )

        answer = (
            f"The column '{column}' contains "
            f"{count:,} unique values."
        )

        return DatasetQAResult(
            success=True,
            question=question,
            operation="unique_values",
            answer=answer,
            evidence={
                "column": column,
                "unique_count": count,
            },
        )

    # ========================================================
    # VALUE COUNTS
    # ========================================================

    def _value_counts(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Return the most frequent values of a column.
        """

        column = self._require_column(
            question
        )

        counts = (
            self.df[
                column
            ]
            .value_counts(
                dropna=False
            )
            .head(10)
        )

        table = (
            counts
            .rename(
                "count"
            )
            .reset_index()
        )

        table.columns = [
            column,
            "count",
        ]

        if table.empty:

            answer = (
                f"No values were found in '{column}'."
            )

        else:

            first_value = (
                table.iloc[0][column]
            )

            first_count = int(
                table.iloc[0]["count"]
            )

            answer = (
                f"The most frequent value in "
                f"'{column}' is '{first_value}', "
                f"appearing {first_count:,} times."
            )

        return DatasetQAResult(
            success=True,
            question=question,
            operation="value_counts",
            answer=answer,
            evidence={
                "column": column,
                "top_values": (
                    table.to_dict(
                        orient="records"
                    )
                ),
            },
            dataframe=table,
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    def _correlation(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Calculate Pearson correlation between two numerical
        columns explicitly mentioned in the question.
        """

        numeric_columns = (
            self.df
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        if len(numeric_columns) < 2:
            raise ValueError(
                "At least two numerical columns are required "
                "for correlation analysis."
            )

        normalized_question = self._normalize_text(
            question
        )
        question_tokens = normalized_question.split()

        def column_is_mentioned(column) -> bool:
            normalized_column = self._normalize_text(
                str(column)
            )

            column_tokens = normalized_column.split()

            if len(column_tokens) == 1:
                token = column_tokens[0]

                if token in question_tokens:
                    return True

                if token.endswith("y"):
                    if (
                        token[:-1] + "ies"
                        in question_tokens
                    ):
                        return True

                elif (
                    token + "s"
                    in question_tokens
                ):
                    return True

                return False

            column_length = len(column_tokens)

            for index in range(
                len(question_tokens) - column_length + 1
            ):
                phrase = " ".join(
                    question_tokens[
                        index:index + column_length
                    ]
                )

                if phrase == normalized_column:
                    return True

            return False

        matched_columns = [
            column
            for column in numeric_columns
            if column_is_mentioned(column)
        ]

        if len(matched_columns) < 2:
            raise ValueError(
                "Please mention two numerical columns whose "
                "correlation you want to calculate."
            )

        column1 = matched_columns[0]
        column2 = matched_columns[1]

        correlation = float(
            self.df[
                [column1, column2]
            ]
            .corr()
            .iloc[0, 1]
        )

        answer = (
            f"The Pearson correlation between "
            f"'{column1}' and '{column2}' is "
            f"{correlation:.4f}."
        )

        return DatasetQAResult(
            success=True,
            question=question,
            operation="correlation",
            answer=answer,
            evidence={
                "column1": column1,
                "column2": column2,
                "correlation": correlation,
                "method": "pearson",
            },
        )

    # ========================================================
    # GROUP-BY
    # ========================================================

    def _groupby(
        self,
        question: str,
    ) -> DatasetQAResult:
        """
        Perform a basic group-by aggregation.

        Example:
        "What is the average salary for each department?"
        """

        normalized = self._normalize_text(
            question
        )

        # ----------------------------------------------------
        # Find aggregation
        # ----------------------------------------------------

        if (
            "sum" in normalized
            or "total" in normalized
        ):
            aggregation = "sum"

        elif "median" in normalized:
            aggregation = "median"

        elif (
            "maximum" in normalized
            or "highest" in normalized
            or "max" in normalized
        ):
            aggregation = "max"

        elif (
            "minimum" in normalized
            or "lowest" in normalized
            or "min" in normalized
        ):
            aggregation = "min"

        else:
            aggregation = "mean"

        # ----------------------------------------------------
        # Find mentioned columns safely
        # ----------------------------------------------------

        question_tokens = normalized.split()

        def column_is_mentioned(column) -> bool:
            normalized_column = self._normalize_text(
                str(column)
            )

            column_tokens = normalized_column.split()

            if len(column_tokens) == 1:

                token = column_tokens[0]

                if token in question_tokens:
                    return True

                if token.endswith("y"):
                    return (
                        token[:-1] + "ies"
                        in question_tokens
                    )

                if token.endswith("s"):
                    return (
                        token[:-1]
                        in question_tokens
                    )

                return (
                    token + "s"
                    in question_tokens
                )

            column_length = len(column_tokens)

            for index in range(
                len(question_tokens) - column_length + 1
            ):

                phrase = " ".join(
                    question_tokens[
                        index:index + column_length
                    ]
                )

                if phrase == normalized_column:
                    return True

            return False

        mentioned_columns = [
            column
            for column in self.columns
            if column_is_mentioned(column)
        ]

        if len(mentioned_columns) < 2:
            raise ValueError(
                "Please mention both the grouping column "
                "and the numerical column."
            )

        # ----------------------------------------------------
        # Identify grouping and aggregation columns
        # ----------------------------------------------------

        group_column = None
        value_column = None

        for column in mentioned_columns:

            if not pd.api.types.is_numeric_dtype(
                self.df[column]
            ):

                if group_column is None:
                    group_column = column

            elif value_column is None:

                value_column = column

        # ----------------------------------------------------
        # If there is no categorical column, use the first
        # mentioned column as the grouping column.
        # ----------------------------------------------------

        if group_column is None:
            group_column = mentioned_columns[0]

        # ----------------------------------------------------
        # Find numerical aggregation column
        # ----------------------------------------------------

        if value_column is None:

            for column in mentioned_columns:

                if column == group_column:
                    continue

                if pd.api.types.is_numeric_dtype(
                    self.df[column]
                ):
                    value_column = column
                    break

        if value_column is None:
            raise ValueError(
                "The aggregation column must be numerical."
            )

        # ----------------------------------------------------
        # Perform group-by aggregation
        # ----------------------------------------------------

        grouped = (
            self.df
            .groupby(
                group_column,
                dropna=False,
            )[value_column]
            .agg(
                aggregation
            )
            .reset_index()
        )

        grouped.columns = [
            group_column,
            value_column,
        ]

        grouped = (
            grouped
            .sort_values(
                value_column,
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        answer = (
            f"The {aggregation} of '{value_column}' "
            f"grouped by '{group_column}' has been calculated."
        )

        return DatasetQAResult(
            success=True,
            question=question,
            operation="groupby",
            answer=answer,
            evidence={
                "group_column": group_column,
                "value_column": value_column,
                "aggregation": aggregation,
                "groups": (
                    grouped.to_dict(
                        orient="records"
                    )
                ),
            },
            dataframe=grouped,
        )



# ============================================================
# CONVENIENCE FUNCTION
# ============================================================


def answer_dataset_question(
    dataframe: pd.DataFrame,
    question: str,
) -> DatasetQAResult:
    """
    Convenience function for answering a dataset question.
    """

    engine = DatasetQA(
        dataframe
    )

    return engine.answer(
        question
    )