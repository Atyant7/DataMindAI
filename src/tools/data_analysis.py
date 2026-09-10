"""
General-purpose deterministic data-analysis tool for DataMindAI.

The LLM decides WHAT analysis is required.

This module decides HOW to execute that analysis safely
against the actual pandas DataFrame.

The LLM should never be responsible for calculating
numerical answers itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


# ============================================================
# RESULT
# ============================================================


@dataclass
class DataAnalysisResult:
    """
    Standard result returned by the data-analysis tool.
    """

    success: bool

    operation: str

    answer: str = ""

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
        Convert the result into a JSON-friendly dictionary.
        """

        result = {
            "success": self.success,
            "operation": self.operation,
            "answer": self.answer,
            "evidence": self.evidence,
            "error": self.error,
            "warnings": self.warnings,
        }

        if self.dataframe is not None:

            result["dataframe"] = (
                self.dataframe
                .to_dict(
                    orient="records"
                )
            )

        else:

            result["dataframe"] = None

        return result


# ============================================================
# DATA ANALYSIS TOOL
# ============================================================


class DataAnalysisTool:
    """
    Execute structured data-analysis operations against
    a pandas DataFrame.

    The LLM supplies structured parameters.

    Example:

    {
        "operation": "groupby",
        "group_column": "company",
        "value_column": "revenue_billions_usd",
        "aggregation": "mean"
    }

    The tool executes that request deterministically.
    """

    SUPPORTED_OPERATIONS = {
        "overview",
        "statistics",
        "groupby",
        "rank",
        "filter",
        "correlation",
        "value_counts",
        "unique_count",
        "missing_values",
        "duplicates",
        "row_extreme",
        "compare",
    }

    SUPPORTED_AGGREGATIONS = {
        "mean",
        "median",
        "sum",
        "min",
        "max",
        "std",
        "count",
    }

    SUPPORTED_OPERATORS = {
        "==",
        "!=",
        ">",
        ">=",
        "<",
        "<=",
    }

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

    # ========================================================
    # PUBLIC EXECUTION API
    # ========================================================

    def execute(
        self,
        operation: str,
        **parameters: Any,
    ) -> DataAnalysisResult:
        """
        Execute one structured analysis operation.
        """

        if not operation:

            return self._error(
                operation="unknown",
                message=(
                    "No analysis operation was provided."
                ),
            )

        operation = (
            operation
            .strip()
            .lower()
        )

        if operation not in self.SUPPORTED_OPERATIONS:

            return self._error(
                operation=operation,
                message=(
                    f"Unsupported analysis operation: "
                    f"'{operation}'."
                ),
            )

        try:

            # LLM tool calls occasionally use the generic ``column`` field
            # for a group-by's numeric value.  Accept that documented alias
            # so an otherwise valid request does not fail with an unexpected
            # keyword-argument error.
            if operation == "groupby":
                parameters = dict(parameters)
                column = parameters.pop("column", None)
                if parameters.get("value_column") is None and column is not None:
                    parameters["value_column"] = column

                # ``order`` is used by rank; ``sort`` is the group-by API.
                # Normalizing it here makes the public tool interface more
                # forgiving without changing the calculation itself.
                order = parameters.pop("order", None)
                if parameters.get("sort") is None and order is not None:
                    parameters["sort"] = order

            if operation == "overview":

                return self._overview()

            if operation == "statistics":

                return self._statistics(
                    **parameters
                )

            if operation == "groupby":

                return self._groupby(
                    **parameters
                )

            if operation == "rank":

                return self._rank(
                    **parameters
                )

            if operation == "filter":

                return self._filter(
                    **parameters
                )

            if operation == "correlation":

                return self._correlation(
                    **parameters
                )

            if operation == "value_counts":

                return self._value_counts(
                    **parameters
                )

            if operation == "unique_count":

                return self._unique_count(
                    **parameters
                )

            if operation == "missing_values":

                return self._missing_values()

            if operation == "duplicates":

                return self._duplicates()

            if operation == "row_extreme":

                return self._row_extreme(
                    **parameters
                )

            if operation == "compare":

                return self._compare(
                    **parameters
                )

            return self._error(
                operation=operation,
                message=(
                    "The requested operation is not implemented."
                ),
            )

        except Exception as exc:

            return self._error(
                operation=operation,
                message=str(exc),
            )

    # ========================================================
    # OVERVIEW
    # ========================================================

    def _overview(
        self,
    ) -> DataAnalysisResult:
        """
        Return a complete high-level dataset overview.
        """

        rows, columns = (
            self.df.shape
        )

        missing_values = int(
            self.df
            .isna()
            .sum()
            .sum()
        )

        duplicate_rows = int(
            self.df
            .duplicated()
            .sum()
        )

        numerical_columns = (
            self.df
            .select_dtypes(
                include="number"
            )
            .columns
            .tolist()
        )

        categorical_columns = (
            self.df
            .select_dtypes(
                include=[
                    "object",
                    "category",
                    "string",
                ]
            )
            .columns
            .tolist()
        )

        evidence = {
            "rows": rows,
            "columns": columns,
            "column_names": [
                str(column)
                for column in self.df.columns
            ],
            "missing_values": missing_values,
            "duplicate_rows": duplicate_rows,
            "numerical_columns": [
                str(column)
                for column in numerical_columns
            ],
            "categorical_columns": [
                str(column)
                for column in categorical_columns
            ],
        }

        answer = (
            f"The dataset contains {rows:,} rows "
            f"and {columns:,} columns. "
            f"It has {missing_values:,} missing values "
            f"and {duplicate_rows:,} duplicate rows."
        )

        return DataAnalysisResult(
            success=True,
            operation="overview",
            answer=answer,
            evidence=evidence,
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    def _statistics(
        self,
        column: str,
        aggregation: str = "mean",
    ) -> DataAnalysisResult:
        """
        Calculate one statistic for a numerical column.
        """

        self._require_column(
            column
        )

        if not pd.api.types.is_numeric_dtype(
            self.df[column]
        ):

            raise ValueError(
                f"Column '{column}' is not numerical."
            )

        aggregation = (
            aggregation
            .strip()
            .lower()
        )

        if aggregation not in self.SUPPORTED_AGGREGATIONS:

            raise ValueError(
                f"Unsupported aggregation '{aggregation}'."
            )

        series = self.df[
            column
        ]

        if aggregation == "mean":

            value = series.mean()

        elif aggregation == "median":

            value = series.median()

        elif aggregation == "sum":

            value = series.sum()

        elif aggregation == "min":

            value = series.min()

        elif aggregation == "max":

            value = series.max()

        elif aggregation == "std":

            value = series.std()

        elif aggregation == "count":

            value = series.count()

        else:

            raise ValueError(
                f"Unsupported aggregation '{aggregation}'."
            )

        if pd.isna(value):

            raise ValueError(
                f"No valid numerical values exist "
                f"in column '{column}'."
            )

        value = float(
            value
        )

        answer = (
            f"The {aggregation} of '{column}' "
            f"is {value:,.4f}."
        )

        return DataAnalysisResult(
            success=True,
            operation="statistics",
            answer=answer,
            evidence={
                "column": column,
                "aggregation": aggregation,
                "value": value,
                "valid_values": int(
                    series.count()
                ),
            },
        )

    # ========================================================
    # GROUP BY
    # ========================================================

    def _groupby(
        self,
        group_column: str,
        value_column: str,
        aggregation: str = "mean",
        limit: int | None = None,
        sort: str = "descending",
    ) -> DataAnalysisResult:
        """
        Group data by one column and aggregate another.
        """

        self._require_column(
            group_column
        )

        self._require_column(
            value_column
        )

        if not pd.api.types.is_numeric_dtype(
            self.df[value_column]
        ):

            raise ValueError(
                f"Column '{value_column}' must be numerical "
                "for aggregation."
            )

        aggregation = (
            aggregation
            .strip()
            .lower()
        )

        if aggregation not in self.SUPPORTED_AGGREGATIONS:

            raise ValueError(
                f"Unsupported aggregation '{aggregation}'."
            )

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

        ascending = (
            sort.lower()
            == "ascending"
        )

        grouped = (
            grouped
            .sort_values(
                value_column,
                ascending=ascending,
            )
            .reset_index(
                drop=True
            )
        )

        if limit is not None:

            limit = int(
                limit
            )

            if limit <= 0:

                raise ValueError(
                    "limit must be greater than zero."
                )

            grouped = grouped.head(
                limit
            )

        records = (
            grouped
            .to_dict(
                orient="records"
            )
        )

        answer = (
            f"Calculated the {aggregation} of "
            f"'{value_column}' grouped by "
            f"'{group_column}'."
        )

        return DataAnalysisResult(
            success=True,
            operation="groupby",
            answer=answer,
            evidence={
                "group_column": group_column,
                "value_column": value_column,
                "aggregation": aggregation,
                "sort": sort,
                "limit": limit,
                "groups": records,
            },
            dataframe=grouped,
        )

    # ========================================================
    # RANKING
    # ========================================================

    def _rank(
        self,
        column: str,
        group_column: str | None = None,
        aggregation: str | None = None,
        order: str = "descending",
        limit: int = 5,
    ) -> DataAnalysisResult:
        """
        Rank rows or groups.

        Examples:

        Top 5 companies by revenue:

            group_column="company"
            column="revenue"
            aggregation="mean"

        Top 5 rows by layoffs:

            group_column=None
            column="layoffs"
        """

        self._require_column(
            column
        )

        limit = int(
            limit
        )

        if limit <= 0:

            raise ValueError(
                "limit must be greater than zero."
            )

        if group_column is not None:

            self._require_column(
                group_column
            )

            if aggregation is None:

                aggregation = "mean"

            if aggregation not in (
                self.SUPPORTED_AGGREGATIONS
            ):

                raise ValueError(
                    f"Unsupported aggregation '{aggregation}'."
                )

            if not pd.api.types.is_numeric_dtype(
                self.df[column]
            ):

                raise ValueError(
                    f"Column '{column}' must be numerical."
                )

            result = (
                self.df
                .groupby(
                    group_column,
                    dropna=False,
                )[column]
                .agg(
                    aggregation
                )
                .reset_index()
            )

        else:

            result = (
                self.df[
                    [column]
                ]
                .copy()
            )

        ascending = (
            order.lower()
            == "ascending"
        )

        result = (
            result
            .sort_values(
                column,
                ascending=ascending,
            )
            .head(
                limit
            )
            .reset_index(
                drop=True
            )
        )

        records = (
            result
            .to_dict(
                orient="records"
            )
        )

        if group_column:

            answer = (
                f"Top {limit} values ranked by "
                f"{aggregation} '{column}' for each "
                f"'{group_column}'."
            )

        else:

            answer = (
                f"Top {limit} rows ranked by "
                f"'{column}'."
            )

        return DataAnalysisResult(
            success=True,
            operation="rank",
            answer=answer,
            evidence={
                "column": column,
                "group_column": group_column,
                "aggregation": aggregation,
                "order": order,
                "limit": limit,
                "ranking": records,
            },
            dataframe=result,
        )

    # ========================================================
    # FILTER
    # ========================================================

    def _filter(
        self,
        column: str,
        operator: str,
        value: Any,
        columns: list[str] | None = None,
        limit: int | None = None,
    ) -> DataAnalysisResult:
        """
        Filter rows using a structured condition.
        """

        self._require_column(
            column
        )

        operator = operator.strip()

        if operator not in self.SUPPORTED_OPERATORS:

            raise ValueError(
                f"Unsupported operator '{operator}'."
            )

        series = self.df[
            column
        ]

        converted_value = (
            self._convert_value(
                series,
                value,
            )
        )

        if operator == "==":

            mask = (
                series
                == converted_value
            )

        elif operator == "!=":

            mask = (
                series
                != converted_value
            )

        elif operator == ">":

            mask = (
                series
                > converted_value
            )

        elif operator == ">=":

            mask = (
                series
                >= converted_value
            )

        elif operator == "<":

            mask = (
                series
                < converted_value
            )

        elif operator == "<=":

            mask = (
                series
                <= converted_value
            )

        else:

            raise ValueError(
                "Invalid filter operator."
            )

        filtered = (
            self.df[
                mask
            ]
            .copy()
        )

        if columns:

            for selected_column in columns:

                self._require_column(
                    selected_column
                )

            filtered = filtered[
                columns
            ]

        if limit is not None:

            limit = int(
                limit
            )

            if limit <= 0:

                raise ValueError(
                    "limit must be greater than zero."
                )

            filtered = filtered.head(
                limit
            )

        records = (
            filtered
            .to_dict(
                orient="records"
            )
        )

        answer = (
            f"Found {len(filtered):,} matching rows "
            f"where '{column}' {operator} "
            f"{converted_value}."
        )

        return DataAnalysisResult(
            success=True,
            operation="filter",
            answer=answer,
            evidence={
                "column": column,
                "operator": operator,
                "value": converted_value,
                "matching_rows": len(
                    filtered
                ),
                "rows": records,
            },
            dataframe=filtered,
        )

    # ========================================================
    # CORRELATION
    # ========================================================

    def _correlation(
        self,
        column1: str,
        column2: str,
        method: str = "pearson",
    ) -> DataAnalysisResult:
        """
        Calculate correlation between two numerical columns.
        """

        self._require_column(
            column1
        )

        self._require_column(
            column2
        )

        if not pd.api.types.is_numeric_dtype(
            self.df[column1]
        ):

            raise ValueError(
                f"Column '{column1}' is not numerical."
            )

        if not pd.api.types.is_numeric_dtype(
            self.df[column2]
        ):

            raise ValueError(
                f"Column '{column2}' is not numerical."
            )

        method = (
            method
            .strip()
            .lower()
        )

        if method not in (
            "pearson",
            "spearman",
            "kendall",
        ):

            raise ValueError(
                "Correlation method must be "
                "pearson, spearman or kendall."
            )

        correlation = (
            self.df[
                [column1, column2]
            ]
            .corr(
                method=method
            )
            .iloc[0, 1]
        )

        if pd.isna(
            correlation
        ):

            raise ValueError(
                "Correlation could not be calculated."
            )

        correlation = float(
            correlation
        )

        answer = (
            f"The {method} correlation between "
            f"'{column1}' and '{column2}' is "
            f"{correlation:.4f}."
        )

        return DataAnalysisResult(
            success=True,
            operation="correlation",
            answer=answer,
            evidence={
                "column1": column1,
                "column2": column2,
                "method": method,
                "correlation": correlation,
            },
        )

    # ========================================================
    # VALUE COUNTS
    # ========================================================

    def _value_counts(
        self,
        column: str,
        limit: int = 10,
    ) -> DataAnalysisResult:
        """
        Return the most frequent values.
        """

        self._require_column(
            column
        )

        limit = int(
            limit
        )

        if limit <= 0:

            raise ValueError(
                "limit must be greater than zero."
            )

        result = (
            self.df[
                column
            ]
            .value_counts(
                dropna=False
            )
            .head(
                limit
            )
            .rename(
                "count"
            )
            .reset_index()
        )

        result.columns = [
            column,
            "count",
        ]

        records = (
            result
            .to_dict(
                orient="records"
            )
        )

        return DataAnalysisResult(
            success=True,
            operation="value_counts",
            answer=(
                f"Calculated the most frequent "
                f"values for '{column}'."
            ),
            evidence={
                "column": column,
                "limit": limit,
                "values": records,
            },
            dataframe=result,
        )

    # ========================================================
    # UNIQUE COUNT
    # ========================================================

    def _unique_count(
        self,
        column: str,
    ) -> DataAnalysisResult:
        """
        Count unique values in a column.
        """

        self._require_column(
            column
        )

        count = int(
            self.df[
                column
            ]
            .nunique(
                dropna=True
            )
        )

        return DataAnalysisResult(
            success=True,
            operation="unique_count",
            answer=(
                f"'{column}' contains "
                f"{count:,} unique values."
            ),
            evidence={
                "column": column,
                "unique_count": count,
            },
        )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    def _missing_values(
        self,
    ) -> DataAnalysisResult:
        """
        Analyze missing values.
        """

        counts = (
            self.df
            .isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        counts = (
            counts[
                counts > 0
            ]
        )

        records = {
            str(column): int(count)
            for column, count
            in counts.items()
        }

        total = int(
            counts.sum()
        )

        return DataAnalysisResult(
            success=True,
            operation="missing_values",
            answer=(
                f"The dataset contains "
                f"{total:,} missing values."
            ),
            evidence={
                "total_missing_values": total,
                "by_column": records,
            },
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    def _duplicates(
        self,
    ) -> DataAnalysisResult:
        """
        Count duplicate rows.
        """

        count = int(
            self.df
            .duplicated()
            .sum()
        )

        percentage = (
            count
            / len(self.df)
            * 100
        )

        return DataAnalysisResult(
            success=True,
            operation="duplicates",
            answer=(
                f"The dataset contains "
                f"{count:,} duplicate rows "
                f"({percentage:.2f}% of all rows)."
            ),
            evidence={
                "duplicate_rows": count,
                "duplicate_percentage": percentage,
            },
        )

    # ========================================================
    # ROW EXTREME
    # ========================================================

    def _row_extreme(
        self,
        column: str,
        direction: str = "max",
        columns: list[str] | None = None,
    ) -> DataAnalysisResult:
        """
        Return the complete row containing the minimum
        or maximum value of a numerical column.

        Example:

        Which year had the highest layoffs?
        """

        self._require_column(
            column
        )

        if not pd.api.types.is_numeric_dtype(
            self.df[column]
        ):

            raise ValueError(
                f"Column '{column}' is not numerical."
            )

        direction = (
            direction
            .strip()
            .lower()
        )

        if direction not in (
            "min",
            "max",
        ):

            raise ValueError(
                "direction must be 'min' or 'max'."
            )

        if direction == "max":

            index = self.df[
                column
            ].idxmax()

        else:

            index = self.df[
                column
            ].idxmin()

        row = (
            self.df
            .loc[
                index
            ]
            .to_dict()
        )

        if columns:

            for selected_column in columns:

                self._require_column(
                    selected_column
                )

            row = {
                selected_column: row[
                    selected_column
                ]
                for selected_column in columns
            }

        value = row[
            column
        ]

        answer = (
            f"The {direction}imum value of "
            f"'{column}' is {value}."
        )

        return DataAnalysisResult(
            success=True,
            operation="row_extreme",
            answer=answer,
            evidence={
                "column": column,
                "direction": direction,
                "value": value,
                "row": row,
            },
        )

    # ========================================================
    # COMPARISON
    # ========================================================

    def _compare(
        self,
        column: str,
        groups: list[Any],
        group_column: str = "company",
        aggregation: str = "mean",
    ) -> DataAnalysisResult:
        """
        Compare selected groups using an aggregation.

        Example:

        Compare AMD and Intel based on revenue.
        """

        self._require_column(
            group_column
        )

        self._require_column(
            column
        )

        if not pd.api.types.is_numeric_dtype(
            self.df[column]
        ):

            raise ValueError(
                f"Column '{column}' must be numerical."
            )

        if not groups:

            raise ValueError(
                "At least one group is required."
            )

        aggregation = (
            aggregation
            .strip()
            .lower()
        )

        if aggregation not in (
            self.SUPPORTED_AGGREGATIONS
        ):

            raise ValueError(
                f"Unsupported aggregation '{aggregation}'."
            )

        filtered = self.df[
            self.df[
                group_column
            ].isin(
                groups
            )
        ]

        if filtered.empty:

            return DataAnalysisResult(
                success=True,
                operation="compare",
                answer=(
                    "No matching records were found "
                    "for the requested groups."
                ),
                evidence={
                    "group_column": group_column,
                    "groups": groups,
                    "rows": [],
                },
            )

        result = (
            filtered
            .groupby(
                group_column,
                dropna=False,
            )[column]
            .agg(
                aggregation
            )
            .reset_index()
        )

        records = (
            result
            .to_dict(
                orient="records"
            )
        )

        return DataAnalysisResult(
            success=True,
            operation="compare",
            answer=(
                f"Compared '{column}' across "
                f"{len(groups)} groups using "
                f"{aggregation}."
            ),
            evidence={
                "group_column": group_column,
                "value_column": column,
                "groups": groups,
                "aggregation": aggregation,
                "comparison": records,
            },
            dataframe=result,
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    def _require_column(
        self,
        column: str,
    ) -> None:
        """
        Validate that a requested column exists.
        """

        if column is None:

            raise ValueError(
                "A column name is required."
            )

        if column not in self.df.columns:

            raise ValueError(
                f"Column '{column}' does not exist "
                "in the dataset."
            )

    @staticmethod
    def _convert_value(
        series: pd.Series,
        value: Any,
    ) -> Any:
        """
        Convert a value to the datatype expected by
        the DataFrame column where possible.
        """

        if value is None:

            return None

        if pd.api.types.is_numeric_dtype(
            series
        ):

            try:

                return float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                raise ValueError(
                    f"Value '{value}' is not numerical."
                )

        return value

    # ========================================================
    # ERROR RESULT
    # ========================================================

    @staticmethod
    def _error(
        operation: str,
        message: str,
    ) -> DataAnalysisResult:
        """
        Create a standardized error result.
        """

        return DataAnalysisResult(
            success=False,
            operation=operation,
            error=message,
        )
