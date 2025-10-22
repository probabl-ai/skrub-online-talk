# %% [markdown]
#
# # Simplified machine learning pipeline

# %%
import skrub

dataset = skrub.datasets.fetch_employee_salaries()
report = skrub.TableReport(dataset.employee_salaries)
report

# %% [markdown]
#
# Let's ask an LLM to come a preprocessing pipeline given the complex dataset that we
# are facing.
#
# **🤖 Prompt:**
#
# *Given the type of data in the `dataset.X` dataframe, can you build a
# preprocessing pipeline that I can then used with scikit-learn pipeline and
# specifically plugging a linear model like the `Ridge` as a predictor.*
#
# *However, you don't need to create the full machine learning pipeline. Only
# the preprocessing stage.*

# %%
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Numerical features preprocessing
numerical_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

# Categorical features preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

# Text features preprocessing (if present)
text_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="")),
        ("tfidf", TfidfVectorizer(max_features=1000, stop_words="english")),
    ]
)

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, make_column_selector(dtype_include=[np.number])),
        (
            "cat",
            categorical_transformer,
            make_column_selector(dtype_include=["object", "category"]),
        ),
        ("text", text_transformer, make_column_selector(dtype_include=["object"])),
    ],
    remainder="passthrough",  # Keep any remaining columns
)

preprocessor

# %%
vectorizer = skrub.TableVectorizer()
vectorizer

# %%
vectorizer.fit_transform(dataset.X)

# %%
vectorizer = skrub.TableVectorizer(
    drop_if_constant=True,
    high_cardinality=skrub.TextEncoder()
)
vectorizer

# %%
from sklearn.linear_model import Ridge

predictive_model = skrub.tabular_pipeline(Ridge())
predictive_model

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    dataset.X, dataset.y, test_size=0.2, random_state=42
)

# %%
predictive_model.fit(X_train, y_train).score(X_test, y_test)

# %%
from sklearn.ensemble import RandomForestRegressor

predictive_model = skrub.tabular_pipeline(RandomForestRegressor())
predictive_model
