import pandas as pd
import plotly.express as px


class VisualizationEngine:

    def __init__(self, dataframe):
        self.df = dataframe

    def generate(self, column1, column2):
        """
        Automatically choose and generate the most appropriate
        visualization for two columns.
        """

        if column1 not in self.df.columns:
            raise ValueError(f"Column '{column1}' does not exist.")

        if column2 not in self.df.columns:
            raise ValueError(f"Column '{column2}' does not exist.")

        df = self.df[[column1, column2]].dropna()

        if df.empty:
            raise ValueError("No usable data available for visualization.")

        col1 = df[column1]
        col2 = df[column2]

        col1_numeric = pd.api.types.is_numeric_dtype(col1)
        col2_numeric = pd.api.types.is_numeric_dtype(col2)

        col1_datetime = pd.api.types.is_datetime64_any_dtype(col1)
        col2_datetime = pd.api.types.is_datetime64_any_dtype(col2)

        unique1 = col1.nunique()
        unique2 = col2.nunique()

        # =========================================================
        # DATETIME + NUMERIC
        # =========================================================

        if col1_datetime and col2_numeric:

            return px.line(
                df.sort_values(column1),
                x=column1,
                y=column2,
                markers=True,
                title=f"{column2} over {column1}"
            )

        if col2_datetime and col1_numeric:

            return px.line(
                df.sort_values(column2),
                x=column2,
                y=column1,
                markers=True,
                title=f"{column1} over {column2}"
            )

        # =========================================================
        # DATETIME + CATEGORICAL
        # =========================================================

        if col1_datetime and not col2_numeric:

            grouped = (
                df.groupby([column1, column2])
                .size()
                .reset_index(name="count")
            )

            return px.line(
                grouped.sort_values(column1),
                x=column1,
                y="count",
                color=column2,
                markers=True,
                title=f"{column2} over time"
            )

        if col2_datetime and not col1_numeric:

            grouped = (
                df.groupby([column2, column1])
                .size()
                .reset_index(name="count")
            )

            return px.line(
                grouped.sort_values(column2),
                x=column2,
                y="count",
                color=column1,
                markers=True,
                title=f"{column1} over time"
            )

        # =========================================================
        # NUMERIC + NUMERIC
        # =========================================================

        if col1_numeric and col2_numeric:

            # Very high number of observations:
            # scatter still represents the relationship best.
            return px.scatter(
                df,
                x=column1,
                y=column2,
                title=f"{column1} vs {column2}",
                opacity=0.65
            )

        # =========================================================
        # CATEGORICAL + NUMERIC
        # =========================================================

        if not col1_numeric and col2_numeric:

            # Few categories → box plot gives distribution
            # information, not just averages.
            if unique1 <= 20:

                return px.box(
                    df,
                    x=column1,
                    y=column2,
                    points="outliers",
                    title=f"{column2} distribution by {column1}"
                )

            # Too many categories → bar chart of averages
            grouped = (
                df.groupby(column1)[column2]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )

            return px.bar(
                grouped,
                x=column1,
                y=column2,
                title=f"Average {column2} by {column1}"
            )

        # =========================================================
        # NUMERIC + CATEGORICAL
        # =========================================================

        if col1_numeric and not col2_numeric:

            if unique2 <= 20:

                return px.box(
                    df,
                    x=column2,
                    y=column1,
                    points="outliers",
                    title=f"{column1} distribution by {column2}"
                )

            grouped = (
                df.groupby(column2)[column1]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )

            return px.bar(
                grouped,
                x=column2,
                y=column1,
                title=f"Average {column1} by {column2}"
            )

        # =========================================================
        # CATEGORICAL + CATEGORICAL
        # =========================================================

        if not col1_numeric and not col2_numeric:

            grouped = (
                df.groupby([column1, column2])
                .size()
                .reset_index(name="count")
            )

            # Too many categories → use top combinations
            if len(grouped) > 30:

                grouped = (
                    grouped
                    .sort_values("count", ascending=False)
                    .head(20)
                )

            return px.bar(
                grouped,
                x=column1,
                y="count",
                color=column2,
                barmode="group",
                title=f"{column1} vs {column2}"
            )

        raise ValueError(
            "Could not determine an appropriate visualization."
        )