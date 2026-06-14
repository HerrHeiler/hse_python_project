from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy import stats
from scipy.stats import f_oneway, kruskal, mannwhitneyu


def st_cats():
    """Custom cats effect"""
    st.markdown(
        """
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; height: 100%; pointer-events: none; z-index: 9999;">
        <img src="https://media.giphy.com/media/JIX9t2j0ZTN9S/giphy.gif" 
             style="position: absolute; bottom: 10%; left: 10%; width: 150px; animation: bounce 2s infinite;">
        <img src="https://media.giphy.com/media/mlvseq9yvZhba/giphy.gif" 
             style="position: absolute; bottom: 10%; right: 10%; width: 150px; animation: bounce 2.5s infinite;">
    </div>
    <style>
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-50px); }
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Student Lifestyle",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_html=True,
)


BASE_DIR = Path.cwd()


@st.cache_data
def load_data():
    raw_path = BASE_DIR / "data" / "raw" / "student_lifestyle_dataset.csv"
    df_raw = pd.read_csv(raw_path)

    new_path = BASE_DIR / "data" / "new_entries" / "new_students.csv"
    if new_path.exists():
        df_new = pd.read_csv(new_path)
        df_raw = pd.concat([df_raw, df_new], ignore_index=True)

    return df_raw


df = load_data()


@st.cache_data
def load_processed_data():
    processed_path = (
        BASE_DIR / "data" / "processed" / "student_lifestyle_with_features.csv"
    )

    if processed_path.exists():
        return pd.read_csv(processed_path)
    else:
        return None


df_processed = load_processed_data()


st.sidebar.title("🐼 Navigation")
section = st.sidebar.radio(
    "Choose a section:",
    [
        "Overview",
        "Data Quality",
        "Outliers Analysis",
        "Distribution Analysis",
        "Categorical Analysis",
        "Descriptive Statistics",
        "Hypothesis Testing",
        "Feature Engineering",
        "API Integration",
    ],
)
st.sidebar.markdown("---")
st.sidebar.info(f"Dataset: {len(df)} rows × {len(df.columns)} columns")

st.markdown(
    '<p class="main-header">🐼 Student Lifestyle Dataset</p>',
    unsafe_allow_html=True,
)
st.markdown(
    "### Exploratory Data Analysis of Student Lifestyle and Academic Performance"
)
st.markdown("---")

# Block 1: Overview
if section == "Overview":
    st.markdown('<p class="sub-header">🐼 Dataset Overview</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", f"{len(df.columns)}")
    col3.metric(
        "Numeric Columns", f"{len(df.select_dtypes(include=[np.number]).columns)}"
    )
    col4.metric(
        "Categorical Columns", f"{len(df.select_dtypes(include=['object']).columns)}"
    )

    st.markdown("### First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### Dataset Structure")

    structure_data = []
    for col in df.columns:
        structure_data.append(
            {
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-Null": df[col].notna().sum(),
                "Null": df[col].isna().sum(),
                "Unique": df[col].nunique(),
            }
        )
    structure_df = pd.DataFrame(structure_data)
    st.dataframe(structure_df, use_container_width=True)

    st.markdown("### Basic Statistics")
    st.dataframe(df.describe().round(3), use_container_width=True)


# Block 2: Data Quality
elif section == "Data Quality":
    st.markdown(
        '<p class="sub-header">🐼 Data Quality Check</p>', unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Missing Values")
        missing = df.isna().sum()
        missing_df = pd.DataFrame(
            {
                "Column": missing.index,
                "Missing Count": missing.values,
                "Missing %": (missing.values / len(df) * 100).round(2),
            }
        )

        if missing.sum() == 0:
            st.success("No missing values found in the dataset.")
        else:
            st.warning(f" There are {missing.sum()} missing values")
            st.dataframe(
                missing_df[missing_df["Missing Count"] > 0], use_container_width=True
            )

    with col2:
        st.markdown("### Duplicates Check")
        total_dups = df.duplicated().sum()
        id_dups = df.duplicated(subset=["Student_ID"]).sum()

        dup_metrics = pd.DataFrame(
            {
                "Check": ["Full Row Duplicates", "Student_ID Duplicates"],
                "Count": [total_dups, id_dups],
                "Status": [
                    "Clean" if total_dups == 0 else " Found",
                    "Clean" if id_dups == 0 else "Found",
                ],
            }
        )
        st.dataframe(dup_metrics, use_container_width=True, hide_index=True)

        if total_dups > 0:
            st.warning(f"Found {total_dups} duplicate rows")
            with st.expander("Show duplicate rows"):
                st.dataframe(df[df.duplicated()], use_container_width=True)

# Block 3: Outliers analysis
elif section == "Outliers Analysis":
    st.markdown(
        '<p class="sub-header">🐼 Outliers Analysis</p>', unsafe_allow_html=True
    )

    st.markdown("Select a column and outlier detection method:")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        outlier_cols = [
            "Study_Hours_Per_Day",
            "Extracurricular_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Social_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day",
            "Grades",
        ]
    selected_col = st.selectbox("Select column:", outlier_cols, index=4)

    with col2:
        st.metric(
            "Column Range",
            f"{df[selected_col].min():.1f} - {df[selected_col].max():.1f}",
        )

    with col3:
        outlier_method = st.radio(
            "Outlier Detection Method:",
            [
                "IQR Method (1.5×IQR)",
                "IQR Method (3.0×IQR)",
                "Percentile (95%)",
                "Percentile (99%)",
                "Custom Threshold",
            ],
            index=0,
        )

    Q1 = df[selected_col].quantile(0.25)
    Q3 = df[selected_col].quantile(0.75)
    IQR = Q3 - Q1

    if outlier_method == "IQR Method (1.5×IQR)":
        threshold = Q3 + 1.5 * IQR
        st.info(
            f"🐼 **IQR Method**: Upper bound = Q3 + 1.5×IQR = {Q3:.2f} + 1.5×{IQR:.2f} = **{threshold:.2f}**"
        )
    elif outlier_method == "IQR Method (3.0×IQR)":
        threshold = Q3 + 3.0 * IQR
        st.info(
            f"🐼 **IQR Method (Conservative)**: Upper bound = Q3 + 3.0×IQR = {Q3:.2f} + 3.0×{IQR:.2f} = **{threshold:.2f}**"
        )
    elif outlier_method == "Percentile (95%)":
        threshold = df[selected_col].quantile(0.95)
        st.info(f"🐼 **95th Percentile**: {threshold:.2f}")
    elif outlier_method == "Percentile (99%)":
        threshold = df[selected_col].quantile(0.99)
        st.info(f"🐼 **99th Percentile**: {threshold:.2f}")
    else:
        calculated_threshold = Q3 + 1.5 * IQR
        max_allowed = float(df[selected_col].max()) * 1.2
        default_value = min(calculated_threshold, float(df[selected_col].max()))

        threshold = st.number_input(
            "Enter custom threshold:",
            min_value=float(df[selected_col].min()),
            max_value=max_allowed,
            value=default_value,
            step=0.1,
            format="%.3f",
        )

        st.warning(
            f"**Custom threshold**: {threshold:.3f} (max allowed: {max_allowed:.2f})"
        )

        if threshold > float(df[selected_col].max()):
            st.info(
                f"Threshold {threshold:.3f} is above max value in data ({df[selected_col].max():.3f})"
            )

    outliers = df[df[selected_col] > threshold]
    df_no_outliers = df[df[selected_col] <= threshold]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", len(df))
    col2.metric("Outliers Found", len(outliers))
    col3.metric("Outlier %", f"{len(outliers) / len(df) * 100:.2f}%")
    col4.metric("Valid Records", len(df_no_outliers))

    if len(outliers) > 0:
        with st.expander(f"View {len(outliers)} outlier statistics"):
            st.write(
                f"**Outlier range:** {threshold:.2f} to {df[selected_col].max():.2f}"
            )
            st.write(
                f"**Outlier values:** Min={outliers[selected_col].min():.2f}, Max={outliers[selected_col].max():.2f}, Mean={outliers[selected_col].mean():.2f}"
            )
            st.dataframe(outliers[[selected_col]].head(20), use_container_width=True)

    mean_with = df[selected_col].mean()
    mean_without = df_no_outliers[selected_col].mean()
    median_with = df[selected_col].median()
    median_without = df_no_outliers[selected_col].median()
    std_with = df[selected_col].std()
    std_without = df_no_outliers[selected_col].std()

    mean_change = (
        abs((mean_with - mean_without) / mean_without * 100) if mean_without != 0 else 0
    )
    median_change = (
        abs((median_with - median_without) / median_without * 100)
        if median_without != 0
        else 0
    )
    std_change = (
        abs((std_with - std_without) / std_without * 100) if std_without != 0 else 0
    )

    st.markdown("### Impact Analysis")

    impact_df = pd.DataFrame(
        {
            "Metric": ["Mean", "Median", "Std Dev"],
            "With Outliers": [
                f"{mean_with:.4f}",
                f"{median_with:.4f}",
                f"{std_with:.4f}",
            ],
            "Without Outliers": [
                f"{mean_without:.4f}",
                f"{median_without:.4f}",
                f"{std_without:.4f}",
            ],
            "Change %": [
                f"{mean_change:.2f}%",
                f"{median_change:.2f}%",
                f"{std_change:.2f}%",
            ],
        }
    )
    st.dataframe(impact_df, use_container_width=True, hide_index=True)

    if mean_change < 1 and median_change < 1 and std_change < 1:
        st.success("🐼 **Impact is MINIMAL** — outliers can be kept in the data")
    elif mean_change < 5 and median_change < 5 and std_change < 5:
        st.warning(
            "🐼 **Impact is MODERATE** — think about analyzing with and without outliers"
        )
    else:
        st.error("🐼 **Impact is STRONG** — outliers strongly distort the analysis")

    st.markdown("### Boxplot Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=df, y=selected_col, ax=axes[0], color="skyblue")
    axes[0].set_title(f"With Outliers (n={len(df)})", fontsize=12, fontweight="bold")
    axes[0].set_ylabel(selected_col.replace("_", " "))
    axes[0].axhline(
        threshold,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Threshold: {threshold:.2f}",
    )
    axes[0].legend()

    sns.boxplot(data=df_no_outliers, y=selected_col, ax=axes[1], color="coral")
    axes[1].set_title(
        f"Without Outliers (n={len(df_no_outliers)})", fontsize=12, fontweight="bold"
    )
    axes[1].set_ylabel(selected_col.replace("_", " "))

    plt.suptitle("Boxplot Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Distribution with outliers highlighted")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.histplot(
        data=df[df[selected_col] <= threshold],
        x=selected_col,
        bins=30,
        kde=True,
        color="skyblue",
        label=f"Valid data (n={len(df_no_outliers)})",
        ax=ax,
    )

    if len(outliers) > 0:
        sns.histplot(
            data=outliers,
            x=selected_col,
            bins=5,
            color="red",
            alpha=0.5,
            label=f"Outliers (n={len(outliers)})",
            ax=ax,
        )

    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold: {threshold:.2f}",
    )
    ax.set_title(
        f"Distribution of {selected_col.replace('_', ' ')}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(selected_col.replace("_", " "))
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    if len(outliers) > 0:
        csv_clean = df_no_outliers.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"Download Dataset Without Outliers ({len(df_no_outliers)} rows)",
            data=csv_clean,
            file_name=f"{selected_col}_no_outliers.csv",
            mime="text/csv",
        )

# Block 4: Distribution analysis
elif section == "Distribution Analysis":
    st.markdown(
        '<p class="sub-header">🐼 Distribution Analysis</p>', unsafe_allow_html=True
    )

    numeric_cols = [
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "Grades",
    ]

    selected_cols = st.multiselect(
        "Select columns to analyze:", numeric_cols, default=numeric_cols
    )

    if selected_cols:
        col1, col2 = st.columns(2)
        with col1:
            bins = st.slider("Number of bins:", 10, 50, 30, 5)
        with col2:
            show_kde = st.checkbox("Show KDE (density curve)", value=True)

        n_cols = len(selected_cols)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, col in enumerate(selected_cols):
            if idx < len(axes):
                sns.histplot(
                    data=df, x=col, bins=bins, kde=show_kde, ax=axes[idx], color="coral"
                )
                axes[idx].set_title(
                    f"Distribution of {col.replace('_', ' ')}",
                    fontsize=11,
                    fontweight="bold",
                )
                axes[idx].set_xlabel(col.replace("_", " "))
                axes[idx].set_ylabel("Count")

                mean_val = df[col].mean()
                axes[idx].axvline(
                    mean_val,
                    color="red",
                    linestyle="--",
                    alpha=0.7,
                    label=f"Mean: {mean_val:.2f}",
                )
                axes[idx].legend()

        for idx in range(len(selected_cols), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Distribution Analysis", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Detailed Statistics")
        for col in selected_cols:
            with st.expander(f"🐼 {col.replace('_', ' ')}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{df[col].mean():.3f}")
                col2.metric("Median", f"{df[col].median():.3f}")
                col3.metric("Std Dev", f"{df[col].std():.3f}")
                col4.metric("Range", f"{df[col].min():.1f} - {df[col].max():.1f}")
    else:
        st.info("Please select at least one column to analyze.")


# Block 5: Categorical analysis
elif section == "Categorical Analysis":
    st.markdown(
        '<p class="sub-header">🐼 Categorical Analysis</p>', unsafe_allow_html=True
    )

    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    selected_cat = st.selectbox("Select categorical column:", categorical_cols, index=0)

    col1, col2 = st.columns(2)
    with col1:
        palette = st.selectbox(
            "Color palette:",
            ["rocket_r", "viridis", "Set2", "coolwarm", "magma"],
            index=0,
        )
    with col2:
        show_percentages = st.checkbox("Show percentages", value=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=df, x=selected_cat, ax=ax, palette=palette)
    ax.set_title(
        f"Distribution of {selected_cat.replace('_', ' ')}",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel(selected_cat.replace("_", " "))
    ax.set_ylabel("Count")

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        if show_percentages:
            percentage = 100 * height / total
            ax.annotate(
                f"{int(height)}\n({percentage:.1f}%)",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        else:
            ax.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Category Statistics")
    cat_stats = df[selected_cat].value_counts().reset_index()
    cat_stats.columns = ["Category", "Count"]
    cat_stats["Percentage"] = (cat_stats["Count"] / total * 100).round(2)
    st.dataframe(cat_stats, use_container_width=True, hide_index=True)


# Block 6: Descriptive statistics
elif section == "Descriptive Statistics":
    st.markdown(
        '<p class="sub-header"> Descriptive Statistics</p>', unsafe_allow_html=True
    )

    numeric_cols = [
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
        "Grades",
    ]

    stats_data = []
    for col in numeric_cols:
        stats_data.append(
            {
                "Field": col.replace("_", " "),
                "Count": df[col].count(),
                "Mean": df[col].mean(),
                "Median": df[col].median(),
                "Std Dev": df[col].std(),
                "Min": df[col].min(),
                "Q1 (25%)": df[col].quantile(0.25),
                "Q3 (75%)": df[col].quantile(0.75),
                "Max": df[col].max(),
                "Range": df[col].max() - df[col].min(),
                "IQR": df[col].quantile(0.75) - df[col].quantile(0.25),
            }
        )

    stats_df = pd.DataFrame(stats_data)

    st.markdown("### Summary Table")
    st.dataframe(stats_df.round(3), use_container_width=True, hide_index=True)

    st.markdown("### Mean vs Median Comparison")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x_pos = np.arange(len(numeric_cols))
    width = 0.35

    means = [df[col].mean() for col in numeric_cols]
    medians = [df[col].median() for col in numeric_cols]

    bars1 = axes[0].bar(
        x_pos - width / 2, means, width, label="Mean", color="skyblue", alpha=0.8
    )
    bars2 = axes[0].bar(
        x_pos + width / 2, medians, width, label="Median", color="coral", alpha=0.8
    )

    axes[0].set_xlabel("Numerical Fields", fontsize=11)
    axes[0].set_ylabel("Value", fontsize=11)
    axes[0].set_title("Mean vs Median Comparison", fontsize=13, fontweight="bold")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(
        [col.replace("_Per_Day", "").replace("_", " ") for col in numeric_cols],
        rotation=45,
        ha="right",
    )
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    for bar in bars2:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    stds = [df[col].std() for col in numeric_cols]
    bars3 = axes[1].bar(x_pos, stds, color="lightgreen", alpha=0.8, edgecolor="green")

    axes[1].set_xlabel("Numerical Fields", fontsize=11)
    axes[1].set_ylabel("Standard Deviation", fontsize=11)
    axes[1].set_title("Standard Deviation by Field", fontsize=13, fontweight="bold")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(
        [col.replace("_Per_Day", "").replace("_", " ") for col in numeric_cols],
        rotation=45,
        ha="right",
    )
    axes[1].grid(True, axis="y", alpha=0.3)

    for bar in bars3:
        height = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.suptitle("Descriptive Statistics Overview", fontsize=15, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Distribution Symmetry Analysis")
    symmetry_data = []
    for col in numeric_cols:
        mean_val = df[col].mean()
        median_val = df[col].median()
        diff = mean_val - median_val
        diff_pct = (diff / median_val) * 100 if median_val != 0 else 0

        if abs(diff_pct) < 5:
            symmetry = "Symmetric (normal-like)"
        elif diff_pct > 5:
            symmetry = "Right-skewed (mean > median)"
        else:
            symmetry = "Left-skewed (mean < median)"

        symmetry_data.append(
            {
                "Field": col.replace("_", " "),
                "Mean": mean_val,
                "Median": median_val,
                "Difference": diff,
                "Diff %": f"{diff_pct:+.2f}%",
                "Distribution": symmetry,
            }
        )

    symmetry_df = pd.DataFrame(symmetry_data)
    st.dataframe(symmetry_df, use_container_width=True, hide_index=True)

    csv = stats_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Descriptive Statistics (CSV)",
        data=csv,
        file_name="descriptive_statistics.csv",
        mime="text/csv",
    )

# Block 7: API integration
elif section == "API Integration":
    st.markdown('<p class="sub-header">🐼 FastAPI REST API</p>', unsafe_allow_html=True)
    try:
        stats_response = requests.get("http://127.0.0.1:8000/statistics", timeout=2)
        api_connected = stats_response.status_code == 200
    except requests.RequestException:
        api_connected = False

    if api_connected:
        st.success("**API Connected** — http://127.0.0.1:8000")

        tab1, tab2, tab3 = st.tabs(["🐼 View Data", "🐼 Add Student", "🐼 Statistics"])

        # method 1: GET /students (overview with filters)
        with tab1:
            st.markdown("### 🐼 Browse Students (GET /students)")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                page = st.number_input("Page", min_value=1, value=1, step=1)
            with col2:
                page_size = st.selectbox("Page Size", [5, 10, 20, 50, 100], index=1)
            with col3:
                gender_filter = st.selectbox("Gender", ["All", "Male", "Female"])
            with col4:
                stress_filter = st.selectbox(
                    "Stress Level", ["All", "Low", "Moderate", "High"]
                )

            col5, col6 = st.columns(2)
            with col5:
                min_grades = st.slider("Min Grades", 0.0, 10.0, 0.0, 0.5)
            with col6:
                max_grades = st.slider("Max Grades", 0.0, 10.0, 10.0, 0.5)

            if st.button("🐼 Load Data", type="primary"):
                params = {"page": page, "page_size": page_size}

                if gender_filter != "All":
                    params["gender"] = gender_filter
                if stress_filter != "All":
                    params["stress_level"] = stress_filter
                if min_grades > 0:
                    params["min_grades"] = min_grades
                if max_grades < 10:
                    params["max_grades"] = max_grades

                try:
                    response = requests.get(
                        "http://127.0.0.1:8000/students", params=params, timeout=5
                    )

                    if response.status_code == 200:
                        data = response.json()

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Records", data["total_records"])
                        col2.metric("Total Pages", data["total_pages"])
                        col3.metric("Current Page", data["current_page"])
                        col4.metric("Page Size", data["page_size"])

                        if data["data"]:
                            df_api = pd.DataFrame(data["data"])
                            st.dataframe(df_api, use_container_width=True)

                            csv = df_api.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv,
                                file_name=f"students_page_{data['current_page']}.csv",
                                mime="text/csv",
                            )
                        else:
                            st.warning("No data found for these filters")
                    else:
                        st.error(f"API Error: {response.status_code}")

                except Exception as e:
                    st.error(f"Request failed: {e}")

        # method 2: POST /students (add new student)
        with tab2:
            st.markdown("###🐼 Add New Student (POST /students)")

            with st.form("add_student_form"):
                col1, col2 = st.columns(2)

                with col1:
                    study = st.number_input("Study Hours/Day", 0.0, 24.0, 7.0, 0.5)
                    extra = st.number_input(
                        "Extracurricular Hours", 0.0, 24.0, 2.0, 0.5
                    )
                    sleep = st.number_input("Sleep Hours", 0.0, 24.0, 7.5, 0.5)
                    social = st.number_input("Social Hours", 0.0, 24.0, 2.0, 0.5)

                with col2:
                    physical = st.number_input("Physical Activity", 0.0, 24.0, 3.0, 0.5)
                    grades = st.number_input("Grades", 0.0, 10.0, 8.0, 0.1)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    stress = st.selectbox("Stress Level", ["Low", "Moderate", "High"])

                submitted = st.form_submit_button("🐼 Add Student", type="primary")

                if submitted:
                    payload = {
                        "Study_Hours_Per_Day": study,
                        "Extracurricular_Hours_Per_Day": extra,
                        "Sleep_Hours_Per_Day": sleep,
                        "Social_Hours_Per_Day": social,
                        "Physical_Activity_Hours_Per_Day": physical,
                        "Stress_Level": stress,
                        "Gender": gender,
                        "Grades": grades,
                    }

                    try:
                        response = requests.post(
                            "http://127.0.0.1:8000/students", json=payload, timeout=5
                        )

                        if response.status_code == 200:
                            st.success(
                                f"**Success** Student added with ID: {response.json()['student_id']}"
                            )
                            st_cats()
                            st.json(response.json())
                        else:
                            st.error(
                                f"Error: {response.json().get('detail', 'Unknown error')}"
                            )

                    except Exception as e:
                        st.error(f"Request failed: {e}")
        st.markdown("---")
        st.markdown("### 🐼 Management")

        if st.button("🐼 Clear all new entries", type="secondary"):
            try:
                response = requests.delete(
                    "http://127.0.0.1:8000/students/clear-new", timeout=5
                )
                if response.status_code == 200:
                    st.success("All new entries cleared")
                    st.info("🐼 Refresh the page to see updated data.")
                else:
                    st.error(f" Error: {response.text}")
            except Exception as e:
                st.error(f"Request failed: {e}")
        # methos 3: GET /statistics (view stats)
        with tab3:
            st.markdown("### 🐼 Dataset Statistics (GET /statistics)")

            if st.button("🐼 Refresh Statistics", type="primary"):
                try:
                    response = requests.get(
                        "http://127.0.0.1:8000/statistics", timeout=5
                    )

                    if response.status_code == 200:
                        stats = response.json()

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Total Students", f"{stats['total_students']:,}")
                        col2.metric("Average Grades", f"{stats['average_grades']:.2f}")
                        col3.metric("Data Points", f"{stats['total_students'] * 9:,}")

                        st.markdown("#### Stress Level Distribution")
                        stress_df = pd.DataFrame(
                            {
                                "Stress Level": list(
                                    stats["stress_distribution"].keys()
                                ),
                                "Count": list(stats["stress_distribution"].values()),
                            }
                        )
                        st.bar_chart(
                            stress_df.set_index("Stress Level"),
                            use_container_width=True,
                        )

                        st.markdown("#### Gender Distribution")
                        gender_df = pd.DataFrame(
                            {
                                "Gender": list(stats["gender_distribution"].keys()),
                                "Count": list(stats["gender_distribution"].values()),
                            }
                        )
                        st.bar_chart(
                            gender_df.set_index("Gender"), use_container_width=True
                        )
                        with st.expander("Raw JSON Response"):
                            st.json(stats)

                except Exception as e:
                    st.error(f"Request failed: {e}")
            else:
                st.info("Click 'Refresh Statistics' to load data from API")

    else:
        st.error("**API is not running**")
        st.markdown("""
        **To start the API:**
        1. Open a new terminal
        2. Navigate to the `apps` folder
        3. Run: `uvicorn api:app --reload --port 8000`
        4. Come back and refresh this page
        """)

        st.code(
            """
        # Terminal commands:
        cd apps
        uvicorn api:app --reload --port 8000
        """,
            language="bash",
        )

# Block 7: Hypothesis
elif section == "Hypothesis Testing":
    st.markdown(
        '<p class="sub-header">🐼 Hypothesis Testing</p>', unsafe_allow_html=True
    )

    hypothesis_tabs = st.tabs(
        [
            "H1: Sleep Groups",
            "H2: Sleep Continuous",
            "H3: Lifestyle Balance",
            "H4: Physical Activity vs Stress",
        ]
    )

    # Hypo 1: Sleep Groups
    with hypothesis_tabs[0]:
        st.markdown("### H1: Students who sleep ≥ 7 hours have higher grades")

        df["Sleep_Group"] = np.where(
            df["Sleep_Hours_Per_Day"] >= 7, ">= 7 hours", "< 7 hours"
        )

        col1, col2 = st.columns(2)

        group_less_7 = df[df["Sleep_Group"] == "< 7 hours"]["Grades"]
        group_7_plus = df[df["Sleep_Group"] == ">= 7 hours"]["Grades"]

        with col1:
            st.metric("Mean (< 7 hours)", f"{group_less_7.mean():.3f}")
            st.metric("Median (< 7 hours)", f"{group_less_7.median():.3f}")
            st.metric("Count", len(group_less_7))

        with col2:
            st.metric("Mean (>= 7 hours)", f"{group_7_plus.mean():.3f}")
            st.metric("Median (>= 7 hours)", f"{group_7_plus.median():.3f}")
            st.metric("Count", len(group_7_plus))

        t_stat, p_val_t = stats.ttest_ind(group_less_7, group_7_plus, equal_var=False)
        u_stat, p_val_mw = mannwhitneyu(
            group_less_7, group_7_plus, alternative="two-sided"
        )

        st.markdown("### Statistical Tests")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**T-Test:**")
            st.write(f"- t-statistic: {t_stat:.3f}")
            st.write(f"- p-value: {p_val_t:.4f}")
            st.success("Significant" if p_val_t < 0.05 else "Not significant")

        with col2:
            st.markdown("**Mann-Whitney U:**")
            st.write(f"- U-statistic: {u_stat:.3f}")
            st.write(f"- p-value: {p_val_mw:.4f}")
            st.success("Significant" if p_val_mw < 0.05 else "Not significant")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.boxplot(
            data=df,
            x="Sleep_Group",
            y="Grades",
            palette={"< 7 hours": "coral", ">= 7 hours": "skyblue"},
            order=["< 7 hours", ">= 7 hours"],
            ax=axes[0],
        )
        axes[0].set_title("Grades by Sleep Group", fontsize=12, fontweight="bold")

        sns.kdeplot(
            data=group_less_7,
            label="< 7 hours",
            color="coral",
            fill=True,
            alpha=0.5,
            ax=axes[1],
        )
        sns.kdeplot(
            data=group_7_plus,
            label=">= 7 hours",
            color="skyblue",
            fill=True,
            alpha=0.5,
            ax=axes[1],
        )
        axes[1].set_title("Density Distribution", fontsize=12, fontweight="bold")
        axes[1].legend()

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Conclusion")
        if p_val_t >= 0.05 and p_val_mw >= 0.05:
            st.warning(
                "**Hypothesis NOT SUPPORTED** — No significant difference between groups"
            )
        else:
            st.success("🐼 Hypothesis supported")

    # Hypo 2: Sleep Continuous
    with hypothesis_tabs[1]:
        st.markdown(
            "### H2: Students with more sleep tend to have higher grades (continuous)"
        )

        pearson_r, pearson_p = stats.pearsonr(df["Sleep_Hours_Per_Day"], df["Grades"])
        spearman_r, spearman_p = stats.spearmanr(
            df["Sleep_Hours_Per_Day"], df["Grades"]
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pearson Correlation:**")
            st.write(f"- r = {pearson_r:.4f}")
            st.write(f"- p-value = {pearson_p:.4f}")

        with col2:
            st.markdown("**Spearman Correlation:**")
            st.write(f"- ρ = {spearman_r:.4f}")
            st.write(f"- p-value = {spearman_p:.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df,
            x="Sleep_Hours_Per_Day",
            y="Grades",
            alpha=0.6,
            s=50,
            color="gray",
            ax=ax,
        )
        sns.regplot(
            data=df,
            x="Sleep_Hours_Per_Day",
            y="Grades",
            scatter=False,
            color="blue",
            ax=ax,
        )
        ax.set_title(
            f"Sleep vs Grades\nPearson r={pearson_r:.3f} (p={pearson_p:.3f})",
            fontsize=13,
            fontweight="bold",
        )
        ax.set_xlabel("Sleep Hours Per Day")
        ax.set_ylabel("Grades")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("### Conclusion")
        if pearson_p >= 0.05:
            st.warning("**Hypothesis NOT SUPPORTED** — No significant correlation")
        else:
            st.success("🐼 Hypothesis supported")

    # Hypo 3: Lifestyle Balance
    with hypothesis_tabs[2]:
        st.markdown("### H3: Students with more balanced lifestyle have higher grades")

        if (
            df_processed is None
            or "Lifestyle_Balance_Score" not in df_processed.columns
        ):
            st.error("""
            **Processed dataset not found**
            
            Please run the Feature Engineering notebook first to create:
            `data/processed/student_lifestyle_with_features.csv`
            
            Or run the Feature Engineering section in this app first.
            """)
        else:
            df_h3 = df_processed.copy()
            df_h3["Balance_Group"] = pd.qcut(
                df_h3["Lifestyle_Balance_Score"],
                q=3,
                labels=["Low Balance", "Medium Balance", "High Balance"],
            )

            st.info(
                f"Using processed dataset with {len(df_h3)} records and {len(df_h3.columns)} columns"
            )

            group_stats = (
                df_h3.groupby("Balance_Group")["Grades"]
                .agg(["count", "mean", "median", "std"])
                .round(3)
            )
            st.markdown("### Descriptive Statistics by Balance Group")
            st.dataframe(group_stats, use_container_width=True)

            # ANOVA, Kruskal-Wallis
            low_balance = df_h3[df_h3["Balance_Group"] == "Low Balance"]["Grades"]
            medium_balance = df_h3[df_h3["Balance_Group"] == "Medium Balance"]["Grades"]
            high_balance = df_h3[df_h3["Balance_Group"] == "High Balance"]["Grades"]

            f_stat, p_val_anova = f_oneway(low_balance, medium_balance, high_balance)
            h_stat, p_val_kw = kruskal(low_balance, medium_balance, high_balance)

            st.markdown("### Statistical Tests")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**ANOVA:**")
                st.write(f"- F-statistic: {f_stat:.3f}")
                st.write(f"- p-value: {p_val_anova:.4f}")
                st.success("Significant" if p_val_anova < 0.05 else "Not significant")

            with col2:
                st.markdown("**Kruskal-Wallis:**")
                st.write(f"- H-statistic: {h_stat:.3f}")
                st.write(f"- p-value: {p_val_kw:.4f}")
                st.success("Significant" if p_val_kw < 0.05 else "Not significant")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(
                data=df_h3,
                x="Balance_Group",
                y="Grades",
                palette=["coral", "skyblue", "lightgreen"],
                order=["Low Balance", "Medium Balance", "High Balance"],
                ax=ax,
            )
            ax.set_title(
                "Grades by Lifestyle Balance Group", fontsize=13, fontweight="bold"
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            pearson_r, pearson_p = stats.pearsonr(
                df_h3["Lifestyle_Balance_Score"], df_h3["Grades"]
            )
            spearman_r, spearman_p = stats.spearmanr(
                df_h3["Lifestyle_Balance_Score"], df_h3["Grades"]
            )

            st.markdown("### Continuous Correlation")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Pearson r:** {pearson_r:.4f} (p={pearson_p:.4f})")
            with col2:
                st.write(f"**Spearman ρ:** {spearman_r:.4f} (p={spearman_p:.4f})")

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(
                data=df_h3,
                x="Lifestyle_Balance_Score",
                y="Grades",
                alpha=0.6,
                s=50,
                color="gray",
                ax=ax,
            )
            sns.regplot(
                data=df_h3,
                x="Lifestyle_Balance_Score",
                y="Grades",
                scatter=False,
                color="blue",
                ax=ax,
            )
            ax.set_title(
                f"Lifestyle Balance Score vs Grades\nPearson r={pearson_r:.3f}",
                fontsize=13,
                fontweight="bold",
            )
            ax.set_xlabel("Lifestyle Balance Score")
            ax.set_ylabel("Grades")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Conclusion")
            if p_val_anova < 0.05:
                low_mean = low_balance.mean()
                high_mean = high_balance.mean()

                if low_mean > high_mean:
                    st.error(f"""
                    **Hypothesis REJECTED (opposite effect)**
                    
                    Less balanced students have **HIGHER** grades:
                    - Low Balance mean: {low_mean:.3f}
                    - High Balance mean: {high_mean:.3f}
                    
                    **Interpretation:** Students who specialize (focus on studying) 
                    outperform those with balanced lifestyles.
                    """)
                else:
                    st.success("🐼 Hypothesis supported")
            else:
                st.warning("Hypothesis not supported")

    # Hypo 4: Physical Activity vs Stress
    with hypothesis_tabs[3]:
        st.markdown(
            "### H4: Among students with similar study hours, higher physical activity → lower stress"
        )

        if (
            df_processed is None
            or "Lifestyle_Balance_Score" not in df_processed.columns
        ):
            st.error("""
            **Processed dataset not found!**
            
            Please run the Feature Engineering notebook first to create:
            `data/processed/student_lifestyle_with_features.csv`
            """)
        else:
            df_h4 = df_processed.copy()

            stress_mapping = {"Low": 1, "Moderate": 2, "High": 3}
            df_h4["Stress_Level_Numeric"] = df_h4["Stress_Level"].map(stress_mapping)

            df_h4["Study_Group"] = pd.qcut(
                df_h4["Study_Hours_Per_Day"],
                q=3,
                labels=["Low Study", "Medium Study", "High Study"],
            )

            st.info(f" Using processed dataset with {len(df_h4)} records")

            r_physical_stress = df_h4["Physical_Activity_Hours_Per_Day"].corr(
                df_h4["Stress_Level_Numeric"]
            )
            r_physical_study = df_h4["Physical_Activity_Hours_Per_Day"].corr(
                df_h4["Study_Hours_Per_Day"]
            )
            r_stress_study = df_h4["Stress_Level_Numeric"].corr(
                df_h4["Study_Hours_Per_Day"]
            )

            partial_r = (
                r_physical_stress - r_physical_study * r_stress_study
            ) / np.sqrt((1 - r_physical_study**2) * (1 - r_stress_study**2))

            st.markdown("### Zero-Order Correlations")
            col1, col2, col3 = st.columns(3)
            col1.metric("Physical vs Stress", f"{r_physical_stress:.4f}")
            col2.metric("Physical vs Study", f"{r_physical_study:.4f}")
            col3.metric("Stress vs Study", f"{r_stress_study:.4f}")

            st.markdown("### Partial Correlation (controlling Study Hours)")
            st.metric("Partial r", f"{partial_r:.4f}")

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            corr_cols = [
                "Study_Hours_Per_Day",
                "Physical_Activity_Hours_Per_Day",
                "Stress_Level_Numeric",
            ]
            corr_matrix = df_h4[corr_cols].corr()
            sns.heatmap(
                corr_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                ax=axes[0],
                fmt=".3f",
                xticklabels=["Study Hours", "Physical Activity", "Stress Level"],
                yticklabels=["Study Hours", "Physical Activity", "Stress Level"],
            )
            axes[0].set_title(
                "Zero-Order Correlation Matrix", fontsize=12, fontweight="bold"
            )

            partial_data = pd.DataFrame(
                {
                    "Type": ["Zero-order", "Partial (controlled)"],
                    "Correlation": [r_physical_stress, partial_r],
                }
            )
            sns.barplot(
                data=partial_data,
                x="Correlation",
                y="Type",
                palette=["coral", "skyblue"],
                ax=axes[1],
            )
            axes[1].axvline(x=0, color="black", linewidth=1)
            axes[1].set_title(
                "Partial Correlation Analysis", fontsize=12, fontweight="bold"
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Physical Activity by Stress Level (within Study Groups)")

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            stress_order = ["Low", "Moderate", "High"]
            colors_stress = ["lightgreen", "orange", "coral"]

            for idx, study_group in enumerate(
                ["Low Study", "Medium Study", "High Study"]
            ):
                subset = df_h4[df_h4["Study_Group"] == study_group]
                ax = axes[idx]

                sns.boxplot(
                    data=subset,
                    x="Stress_Level",
                    y="Physical_Activity_Hours_Per_Day",
                    palette=colors_stress,
                    order=stress_order,
                    width=0.5,
                    ax=ax,
                )
                ax.set_title(
                    f"{study_group} (n={len(subset)})", fontsize=11, fontweight="bold"
                )
                ax.set_xlabel("Stress Level")
                ax.set_ylabel("Physical Activity Hours")

            plt.suptitle(
                "Physical Activity by Stress Level (Controlling for Study Hours)",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("### Conclusion")
            if partial_r > 0 and abs(partial_r) > 0.1:
                st.error(
                    """
                **Hypothesis REJECTED (opposite effect)**
                
                Among students with similar study hours, those with **higher** physical activity 
                have **HIGHER** stress levels (partial r = {partial_r:.4f}).
                
                **This is Simpson's Paradox**🐼🐼🐼
                - Without controlling Study Hours: r = {r_physical_stress:.4f} (negative)
                - After controlling Study Hours: r = {partial_r:.4f} (positive)
                
                **Interpretation:** Students who add physical activity to already high study 
                loads may experience increased stress due to time pressure.
                """.format(partial_r=partial_r, r_physical_stress=r_physical_stress)
                )
            elif partial_r < 0 and abs(partial_r) > 0.1:
                st.success("🐼 Hypothesis supported")
            else:
                st.warning("Hypothesis not supported — no meaningful relationship")

# Block 8: features
elif section == "Feature Engineering":
    st.markdown(
        '<p class="sub-header">🔧 Feature Engineering</p>', unsafe_allow_html=True
    )

    if "Total_Active_Hours" not in df.columns:
        df["Total_Active_Hours"] = (
            df["Study_Hours_Per_Day"]
            + df["Extracurricular_Hours_Per_Day"]
            + df["Social_Hours_Per_Day"]
            + df["Physical_Activity_Hours_Per_Day"]
        )

        df["Social_Share"] = df["Social_Hours_Per_Day"] / df["Total_Active_Hours"]
        df["Study_to_Sleep_Ratio"] = (
            df["Study_Hours_Per_Day"] / df["Sleep_Hours_Per_Day"]
        )
        df["Physical_to_Study_Ratio"] = (
            df["Physical_Activity_Hours_Per_Day"] / df["Study_Hours_Per_Day"]
        )

        activity_cols = [
            "Study_Hours_Per_Day",
            "Extracurricular_Hours_Per_Day",
            "Social_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day",
        ]
        df["Lifestyle_Balance"] = df[activity_cols].std(axis=1)
        df["Lifestyle_Balance_Score"] = 1 / (1 + df["Lifestyle_Balance"])

    st.markdown("### New Features Created")

    features_info = {
        "Total_Active_Hours": "Sum of all active hours (study + extracurricular + social + physical)",
        "Social_Share": "Proportion of social time in total active time",
        "Study_to_Sleep_Ratio": "Ratio of study hours to sleep hours",
        "Physical_to_Study_Ratio": "Ratio of physical activity to study hours",
        "Lifestyle_Balance_Score": "Balance score (higher = more balanced lifestyle)",
    }

    for feature, description in features_info.items():
        with st.expander(f"🐼 {feature}"):
            st.write(f"**Description:** {description}")
            st.write(f"**Mean:** {df[feature].mean():.3f}")
            st.write(f"**Median:** {df[feature].median():.3f}")
            st.write(f"**Std:** {df[feature].std():.3f}")
            st.write(f"**Range:** [{df[feature].min():.3f}, {df[feature].max():.3f}]")

    st.markdown("### Correlation with Grades")

    new_features = [
        "Social_Share",
        "Study_to_Sleep_Ratio",
        "Physical_to_Study_Ratio",
        "Lifestyle_Balance_Score",
    ]

    correlation_data = []
    for feature in new_features:
        pearson_r, pearson_p = stats.pearsonr(df[feature], df["Grades"])
        spearman_r, spearman_p = stats.spearmanr(df[feature], df["Grades"])

        correlation_data.append(
            {
                "Feature": feature,
                "Pearson r": f"{pearson_r:.4f}",
                "Pearson p": f"{pearson_p:.4f}",
                "Spearman ρ": f"{spearman_r:.4f}",
                "Spearman p": f"{spearman_p:.4f}",
            }
        )

    corr_df = pd.DataFrame(correlation_data)
    st.dataframe(corr_df, use_container_width=True, hide_index=True)

    st.markdown("### Feature Distributions")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(df["Social_Share"], bins=30, kde=True, ax=axes[0, 0], color="skyblue")
    axes[0, 0].set_title("Social Share Distribution", fontsize=12, fontweight="bold")

    sns.histplot(
        df["Study_to_Sleep_Ratio"], bins=30, kde=True, ax=axes[0, 1], color="coral"
    )
    axes[0, 1].set_title(
        "Study-to-Sleep Ratio Distribution", fontsize=12, fontweight="bold"
    )

    sns.histplot(
        df["Physical_to_Study_Ratio"],
        bins=30,
        kde=True,
        ax=axes[1, 0],
        color="lightgreen",
    )
    axes[1, 0].set_title(
        "Physical-to-Study Ratio Distribution", fontsize=12, fontweight="bold"
    )

    sns.histplot(
        df["Lifestyle_Balance_Score"], bins=30, kde=True, ax=axes[1, 1], color="orange"
    )
    axes[1, 1].set_title(
        "Lifestyle Balance Score Distribution", fontsize=12, fontweight="bold"
    )

    plt.suptitle("New Features Distributions", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Study-to-Sleep Ratio vs Grades")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="Study_to_Sleep_Ratio",
        y="Grades",
        alpha=0.6,
        s=50,
        color="gray",
        ax=ax,
    )
    sns.regplot(
        data=df,
        x="Study_to_Sleep_Ratio",
        y="Grades",
        scatter=False,
        color="blue",
        ax=ax,
    )
    ax.set_title("Study-to-Sleep Ratio vs Grades", fontsize=14, fontweight="bold")
    ax.set_xlabel("Study Hours / Sleep Hours")
    ax.set_ylabel("Grades")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### Key Insights")

    new_features = [
        "Social_Share",
        "Study_to_Sleep_Ratio",
        "Physical_to_Study_Ratio",
        "Lifestyle_Balance_Score",
    ]

    correlations = []
    for feature in new_features:
        pearson_r, _ = stats.pearsonr(df[feature], df["Grades"])
        correlations.append(
            {"feature": feature, "r": pearson_r, "abs_r": abs(pearson_r)}
        )

    correlations.sort(key=lambda x: x["abs_r"], reverse=True)

    def get_insight_style(r, abs_r):
        if abs_r >= 0.4:
            if r > 0:
                return "success"
            else:
                return "error"
        elif abs_r >= 0.2:
            if r > 0:
                return "info"
            else:
                return "warning"
        else:
            return "info"

    for corr in correlations:
        feature = corr["feature"]
        r = corr["r"]
        abs_r = corr["abs_r"]
        style = get_insight_style(r, abs_r)
        if abs_r >= 0.4:
            strength = "strongest" if abs_r == correlations[0]["abs_r"] else "strong"
            direction = "positive" if r > 0 else "negative"
            insight = f"**{feature}** is a {strength} predictor (r = {r:.3f})"
            if feature == "Lifestyle_Balance_Score" and r < 0:
                insight += " — focused on study students perform better"
        elif abs_r >= 0.2:
            direction = "positive" if r > 0 else "negative"
            insight = f"**{feature}** shows {direction} correlation (r = {r:.3f})"
        else:
            insight = f"**{feature}** has minimal impact (r = {r:.3f})"
        if style == "success":
            st.success(f"{insight}")
        elif style == "error":
            st.error(f"{insight}")
        elif style == "warning":
            st.warning(f"{insight}")
        else:
            st.info(f"{insight}")


st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: white;'>🐼 Student Lifestyle</div>",
    unsafe_allow_html=True,
)
