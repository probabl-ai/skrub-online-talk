# %% [markdown]
#
# # Additional preprocessing techniques

# %%
import skrub

dataset = skrub.datasets.fetch_employee_salaries()
report = skrub.TableReport(dataset.employee_salaries)
report

# %% [markdown]
#
# Scikit-learn already provides a range of transformers that can be used for
# preprocessing data ahead of machine learning models.

# %%
from sklearn import preprocessing
from sklearn.base import BaseEstimator

print("Transformer used for preprocesing in scikit-learn\n")
for name in dir(preprocessing):
    if isinstance(obj := getattr(preprocessing, name), type) and issubclass(
        obj, BaseEstimator
    ):
        print(f"\t{name}")

# %% [markdown]
#
# However, it does not cover some of the use cases that are not necessarily in the
# scope of scikit-learn.
#
# ## Data cleaning

# %%
import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        "Int": [2, 3, 2],  # Multiple unique values
        "Const str": ["x", "x", "x"],  # Single unique value
        "Str": ["foo", "bar", "baz"],  # Multiple unique values
        "All nan": [np.nan, np.nan, np.nan],  # All missing values
        "All empty": ["", "", ""],  # All empty strings
        "Date": ["01 Jan 2023", "02 Jan 2023", "03 Jan 2023"],
    }
)
df

# %% [markdown]
#
# What an LLM would do to prepare such dataframe for machine learning?
#
# Prompt:
#
# Given the following dataframe, can you write some pandas code to clean the dataframe.

# %%
# Step 1: Remove columns with all missing values (NaN)
df_cleaned = df.dropna(axis=1, how="all")
df_cleaned

# %%
# Step 2: Remove columns with all empty strings
df_cleaned = df_cleaned.loc[:, ~(df_cleaned == "").all()]
df_cleaned

# %%
# Step 3: Remove constant columns (columns with only one unique value)
constant_columns = []
for col in df_cleaned.columns:
    if df_cleaned[col].nunique() <= 1:
        constant_columns.append(col)

df_cleaned = df_cleaned.drop(columns=constant_columns)
df_cleaned

# %%
# Step 4: Handle remaining missing values

# For numerical columns, fill with median
numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df_cleaned[col].isnull().any():
        median_val = df_cleaned[col].median()
        df_cleaned[col] = df_cleaned[col].fillna(median_val)

# For categorical columns, fill with mode or 'Unknown'
categorical_cols = df_cleaned.select_dtypes(include=["object"]).columns
for col in categorical_cols:
    if df_cleaned[col].isnull().any():
        mode_val = df_cleaned[col].mode()
        if len(mode_val) > 0:
            df_cleaned[col] = df_cleaned[col].fillna(mode_val[0])
        else:
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

df_cleaned

# %%
# Step 5: Clean string columns (remove extra whitespace, handle empty strings)
for col in categorical_cols:
    if col in df_cleaned.columns:
        # Strip whitespace
        df_cleaned[col] = df_cleaned[col].astype(str).str.strip()
        # Replace empty strings with NaN, then fill with 'Unknown'
        df_cleaned[col] = df_cleaned[col].replace("", np.nan)
        if df_cleaned[col].isnull().any():
            df_cleaned[col] = df_cleaned[col].fillna("Unknown")

df_cleaned

# %%
# Step 6: Convert date column to proper datetime format
if "Date" in df_cleaned.columns:
    try:
        df_cleaned["Date"] = pd.to_datetime(df_cleaned["Date"], format="%d %b %Y")
        print("Converted Date column to datetime format")
    except:
        print("Could not convert Date column to datetime format")

df_cleaned

# %%
cleaner = skrub.Cleaner(drop_if_constant=True, n_jobs=-1)
cleaner.fit_transform(df)

# %% [markdown]
#
# **Aside note**: Polars dataframe also works out-of-the-box.

# %%
import polars as pl

cleaner.fit_transform(pl.from_dataframe(df))

# %% [markdown]
#
# ## Date and time feature encoding

# %%
df = pd.DataFrame(
    {
        "date": ["2023-01-01 12:34:56", "2023-02-15 08:45:23", "2023-03-20 18:12:45"],
        "value": [10, 20, 30],
    }
)
df

# %% [markdown]
#
# What strategy an LLM would do to encode such date and time features?
#
# Prompt:
#
# I'm doing some machine learning using the following data containing date. Could you
# make some processing such that I can use a linear model later.

# %%
# Convert date column to datetime format
df_encoded = df.copy()
df_encoded["date"] = pd.to_datetime(df_encoded["date"])
df_encoded

# %%
# Extract basic temporal features
df_encoded["year"] = df_encoded["date"].dt.year
df_encoded["month"] = df_encoded["date"].dt.month
df_encoded["day"] = df_encoded["date"].dt.day
df_encoded["hour"] = df_encoded["date"].dt.hour
df_encoded["minute"] = df_encoded["date"].dt.minute
df_encoded["second"] = df_encoded["date"].dt.second

# Extract additional time-based features
df_encoded["day_of_week"] = df_encoded["date"].dt.dayofweek  # 0=Monday, 6=Sunday
df_encoded["day_of_year"] = df_encoded["date"].dt.dayofyear
df_encoded["week_of_year"] = df_encoded["date"].dt.isocalendar().week
df_encoded["quarter"] = df_encoded["date"].dt.quarter

df_encoded

# %%
# Create cyclical features for time components
# These are important for linear models as they capture the cyclical nature of time

# Cyclical encoding for month (1-12)
df_encoded["month_sin"] = np.sin(2 * np.pi * df_encoded["month"] / 12)
df_encoded["month_cos"] = np.cos(2 * np.pi * df_encoded["month"] / 12)

# Cyclical encoding for day of week (0-6)
df_encoded["day_of_week_sin"] = np.sin(2 * np.pi * df_encoded["day_of_week"] / 7)
df_encoded["day_of_week_cos"] = np.cos(2 * np.pi * df_encoded["day_of_week"] / 7)

# Cyclical encoding for hour (0-23)
df_encoded["hour_sin"] = np.sin(2 * np.pi * df_encoded["hour"] / 24)
df_encoded["hour_cos"] = np.cos(2 * np.pi * df_encoded["hour"] / 24)

# Cyclical encoding for day of year (1-365/366)
df_encoded["day_of_year_sin"] = np.sin(2 * np.pi * df_encoded["day_of_year"] / 365.25)
df_encoded["day_of_year_cos"] = np.cos(2 * np.pi * df_encoded["day_of_year"] / 365.25)

df_encoded

# %%
# Create time-based categorical features
df_encoded["is_weekend"] = (df_encoded["day_of_week"] >= 5).astype(int)
df_encoded["is_weekday"] = (df_encoded["day_of_week"] < 5).astype(int)

# Time of day categories
df_encoded["time_of_day"] = pd.cut(
    df_encoded["hour"],
    bins=[0, 6, 12, 18, 24],
    labels=["night", "morning", "afternoon", "evening"],
    include_lowest=True,
)


# Season based on month
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "autumn"


df_encoded["season"] = df_encoded["month"].apply(get_season)

df_encoded

# %%
from sklearn.pipeline import make_pipeline

encoder = make_pipeline(
    skrub.ToDatetime(), skrub.DatetimeEncoder(periodic_encoding="circular")
)
encoder.fit_transform(df["date"])

# %% [markdown]
#
# ## Encoding of categorical features

# %%
df = dataset.employee_salaries[
    [
        "gender",
        "department",
        "department_name",
        "division",
        "assignment_category",
        "employee_position_title",
    ]
]
skrub.TableReport(df)

# %% [markdown]
#
# Let's ask an LLM to check what would make sense to encode categorical features.
#
# Prompt:
#
# Given such categories, what strategy would you use to encode low cardinality
# feature such "gender" and high cardinality feature such as "division".

# %% [markdown]
#
# **Summary of encoding strategies**
#
# For LOW CARDINALITY features (≤ 10 unique values):
# - **One-Hot Encoding:** Best for nominal categories without inherent order
# - **Label Encoding:** Simple but can introduce artificial ordering
#
# For HIGH CARDINALITY features (> 10 unique values):
# - **Target Encoding:** Excellent when you have a target variable, handles overfitting
# - **Frequency Encoding:** Simple, preserves information about category frequency
# - **Ordinal Encoding:** Good when categories have meaningful order
# - **GapEncoder:** Advanced method that learns dense representations
# - **MinHashEncoder:** Good for very high cardinality, approximate but fast
#
# RECOMMENDATIONS:
# 1. `gender` (2 values): One-Hot Encoding
# 2. `assignment_category` (2 values): One-Hot Encoding
# 3. `department` (37 values): Target Encoding or Frequency Encoding
# 4. `division` (694 values): Target Encoding or GapEncoder
# 5. `employee_position_title` (443 values): Target Encoding or GapEncoder

# %%
encoder = skrub.StringEncoder()
encoder.fit_transform(df["division"])

# %%
encoder = skrub.TextEncoder(device="mps")
encoder.fit_transform(df["division"])
