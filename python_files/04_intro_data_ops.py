# %% [markdown]
#
# # Machine learning pipelines with multiple tables

# %%
import skrub

dataset = skrub.datasets.fetch_credit_fraud(split="all")

# %%
skrub.TableReport(dataset.baskets)

# %%
skrub.TableReport(dataset.products)

# %% [markdown]
#
# ## The current state
#
# Let's ask an LLM to create a machine learning pipeline for the given dataset that is
# composed of two tables.
#
# **🤖 Prompt:**
#
# *I would like you to write a machine learning pipeline.
#
# The dataset is composed of two tables: `dataset.baskets` and `data.product`. The
# variable to predict is the column "fraud_flag" from the `data.baskets` table.
#
# So to create the a machine learning pipeline, you will need to kind of join the
# information between the two table and train and test a machine learning model for it.*

# %%
# Join the tables to create features
# First, let's aggregate product information by basket
basket_features = (
    dataset.products.groupby("basket_ID")
    .agg(
        {
            "cash_price": ["sum", "mean", "std", "min", "max", "count"],
            "Nbr_of_prod_purchas": ["sum", "mean", "std"],
            "item": "nunique",  # Number of unique items
            "make": "nunique",  # Number of unique makes
            "model": "nunique",  # Number of unique models
            "goods_code": "nunique",  # Number of unique goods codes
        }
    )
    .reset_index()
)

# Flatten column names
basket_features.columns = ["basket_ID"] + [
    f"{col[0]}_{col[1]}" for col in basket_features.columns[1:]
]

# Add additional features
basket_features["avg_price_per_item"] = (
    basket_features["cash_price_sum"] / basket_features["Nbr_of_prod_purchas_sum"]
)
basket_features["price_std_norm"] = basket_features["cash_price_std"] / (
    basket_features["cash_price_mean"] + 1e-8
)

# %%
skrub.TableReport(basket_features)

# %%
# Merge with fraud labels
df = dataset.baskets.merge(
    basket_features, right_on="basket_ID", left_on="ID", how="left"
)
skrub.TableReport(df)

# %%
# Create additional features for fraud detection
# These are domain-specific features that might be indicative of fraud

# 1. Price anomaly features
df["price_anomaly"] = (
    df["cash_price_sum"] > df["cash_price_sum"].quantile(0.95)
).astype(int)
df["low_price_anomaly"] = (
    df["cash_price_sum"] < df["cash_price_sum"].quantile(0.05)
).astype(int)

# 2. Quantity anomaly features
df["quantity_anomaly"] = (
    df["Nbr_of_prod_purchas_sum"] > df["Nbr_of_prod_purchas_sum"].quantile(0.95)
).astype(int)

# 3. Diversity features
df["item_diversity"] = df["item_nunique"] / (df["Nbr_of_prod_purchas_sum"] + 1e-8)
df["make_diversity"] = df["make_nunique"] / (df["Nbr_of_prod_purchas_sum"] + 1e-8)

# 4. Price consistency features
df["price_consistency"] = 1 / (
    df["cash_price_std"] + 1e-8
)  # Higher values = more consistent prices

# %%
skrub.TableReport(df)

# %%
# Split the data to be aligned with the next example
id_split = 76543  # noqa
df_train = df.query("ID <= @id_split")
df_test = df.query("ID > @id_split")

X_train = df_train.drop(["basket_ID", "ID", "fraud_flag"], axis=1)
y_train = df_train["fraud_flag"]
X_test = df_test.drop(["basket_ID", "ID", "fraud_flag"], axis=1)
y_test = df_test["fraud_flag"]

# %%
# Fill missing values
X_train = X_train.fillna(0)  # Fill NaN with 0 for numerical features
X_test = X_test.fillna(0)

# %%
# Create preprocessing pipeline using skrub's TableVectorizer
preprocessor = skrub.TableVectorizer(
    drop_if_constant=True,
    high_cardinality=skrub.TextEncoder(),
    low_cardinality=skrub.StringEncoder(),
)


# %%
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# Train multiple models and compare performance
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=42, max_iter=100),
}

# %%
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Create pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", model)])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    # Calculate metrics
    auc_score = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        "pipeline": pipeline,
        "predictions": y_pred,
        "probabilities": y_pred_proba,
        "auc_score": auc_score,
    }

    print(f"{name} - AUC Score: {auc_score:.4f}")

# %%
from sklearn.metrics import classification_report, confusion_matrix

# Evaluate the best model (Random Forest typically performs well on tabular data)
best_model_name = "Random Forest"
best_pipeline = results[best_model_name]["pipeline"]
y_pred = results[best_model_name]["predictions"]
y_pred_proba = results[best_model_name]["probabilities"]

print(f"Best Model: {best_model_name}")
print(f"AUC Score: {results[best_model_name]['auc_score']:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve

# Visualize results
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Fraud Detection Model Evaluation", fontsize=16, fontweight="bold")

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0])
axes[0, 0].set_title("Confusion Matrix")
axes[0, 0].set_xlabel("Predicted")
axes[0, 0].set_ylabel("Actual")

# 2. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0, 1].plot(
    fpr,
    tpr,
    linewidth=2,
    label=f'ROC Curve (AUC = {results[best_model_name]["auc_score"]:.3f})',
)
axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
axes[0, 1].set_xlabel("False Positive Rate")
axes[0, 1].set_ylabel("True Positive Rate")
axes[0, 1].set_title("ROC Curve")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Feature Importance (for Random Forest)
if hasattr(best_pipeline.named_steps["classifier"], "feature_importances_"):
    feature_names = best_pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = best_pipeline.named_steps["classifier"].feature_importances_

    # Get top 10 most important features
    top_indices = np.argsort(importances)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = importances[top_indices]

    axes[1, 0].barh(range(len(top_features)), top_importances)
    axes[1, 0].set_yticks(range(len(top_features)))
    axes[1, 0].set_yticklabels(top_features)
    axes[1, 0].set_xlabel("Feature Importance")
    axes[1, 0].set_title("Top 10 Most Important Features")

# 4. Model Comparison
model_names = list(results.keys())
auc_scores = [results[name]["auc_score"] for name in model_names]
axes[1, 1].bar(model_names, auc_scores, color=["skyblue", "lightgreen", "lightcoral"])
axes[1, 1].set_ylabel("AUC Score")
axes[1, 1].set_title("Model Comparison")
axes[1, 1].tick_params(axis="x", rotation=45)
for i, v in enumerate(auc_scores):
    axes[1, 1].text(i, v + 0.001, f"{v:.3f}", ha="center", va="bottom")


# %% [markdown]
#
# **Pitfalls:**
# - Data leakage
# - Headache to replay the same code for a new batch of fresh data
# - Hard to tune hyperparameters related to the preprocessing steps

# %% [markdown]
#
# ## The `skrub` Data Operations (DataOps)
#
# DataOps extend the scikit-learn machinery to allow more complex data operations but
# taking care about machine learning states (i.e. fit / predict).

# %%
dataset = skrub.datasets.fetch_credit_fraud(split="train")
baskets_experiment, products_experiment = dataset.baskets, dataset.products
dataset = skrub.datasets.fetch_credit_fraud(split="test")
baskets_production, products_production = dataset.baskets, dataset.products

# %% [markdown]
#
# To achieve the traceability of the data operations, `skrub` records them. We call
# this a DataOps plan and it is a graph of the data operations. We can record several
# type of operations:
#
# - any dataframe operation (e.g. merge, group by, aggregate, etc.)
# - scikit-learn estimators (e.g. `StandardScaler`, `LogisticRegression`, etc.)
# - any custom user code (e.g. load data from path, any data transformation, etc.)
#
# Everything starts from a `skrub` variable that is a symbolic representation of the
# data.

# %%
products = skrub.var("products")
products

# %% [markdown]
#
# There is not much for the moment. But what if we would like to apply a transformation.
# We can take the previous transform that the LLM created to aggregate statistics for
# a given basket.
#
# The trick here is that we don't want to evaluate the function directly but record
# that we should apply it. Therefore, we use the `@skrub.deferred` decorator.


# %%
@skrub.deferred
def aggregate_basket_features(products):
    basket_features = (
        products.groupby("basket_ID")
        .agg(
            {
                "cash_price": ["sum", "mean", "std", "min", "max", "count"],
                "Nbr_of_prod_purchas": ["sum", "mean", "std"],
                "item": "nunique",  # Number of unique items
                "make": "nunique",  # Number of unique makes
                "model": "nunique",  # Number of unique models
                "goods_code": "nunique",  # Number of unique goods codes
            }
        )
        .reset_index()
    )

    # Flatten column names
    basket_features.columns = ["basket_ID"] + [
        f"{col[0]}_{col[1]}" for col in basket_features.columns[1:]
    ]

    # Add additional features
    basket_features["avg_price_per_item"] = (
        basket_features["cash_price_sum"] / basket_features["Nbr_of_prod_purchas_sum"]
    )
    basket_features["price_std_norm"] = basket_features["cash_price_std"] / (
        basket_features["cash_price_mean"] + 1e-8
    )

    return basket_features


# %%
basket_features = aggregate_basket_features(products)
basket_features

# %% [markdown]
#
# Applying the function do not execute anything but is adding a new node in the DataOps
# plan. However, you can imagine that building a complex DataOps plan without having
# eager feedback would not be practical.
#
# It is one reason, that you can attach some data to the `skrub` variable to compute
# a preview on this data.

# %%
products = skrub.var("products", products_experiment)
products

# %%
basket_features = aggregate_basket_features(products)
basket_features

# %% [markdown]
#
# Now, you have both the graph and the preview of the transformation on your data.
#
# We can do the same on the baskets table.

# %%
baskets = skrub.var("baskets", baskets_experiment)
baskets

# %% [markdown]
#
# However, here we see that the preview is done on the full training set. We might be
# interested in just having the preview on a subsample of the data at least for the
# debugging phase when building our DataOps plan. It is the job of the `skb.subsample`
# method to do this.

# %%
baskets = baskets.skb.subsample(n=5_000)
baskets

# %% [markdown]
#
# When we previously discussed pitfalls of the LLM generated code, we mentioned issue
# related to data leakage. Usually, this problem boils down to not compute statistics
# on the training set and apply them on the test set and instead use the full dataset.
#
# Therefore, we introduce the `skb.mark_as_X` and `skb.mark_as_y` methods to mark nodes
# in the DataOps plan as the features and target variables. It informs that any
# subsequent machine learning operations relying on training / testing sets should
# start from the marked nodes. In short, if a split should happen, it should be done
# at the marked nodes.

# %%
features = baskets[["ID"]].skb.mark_as_X()
target = baskets["fraud_flag"].skb.mark_as_y()

# %%
features

# %%
target

# %% [markdown]
#
# Now, we are going to apply the different operations that the LLM came with. First, we
# need to join the baskets and the aggregated products features.


# %%
@skrub.deferred
def join_basket_aggregated_products(baskets, basket_features):
    return baskets.merge(
        basket_features, right_on="basket_ID", left_on="ID", how="left"
    ).drop(columns=["basket_ID", "ID"])


# %%
aggregated_features = join_basket_aggregated_products(features, basket_features)
aggregated_features

# %% [markdown]
#
# Then, some domain-specific features have been derived from the aggregated features.
# The original code was the following:
#
# ```python
# # 1. Price anomaly features
# df["price_anomaly"] = (
#     df["cash_price_sum"] > df["cash_price_sum"].quantile(0.95)
# ).astype(int)
# df["low_price_anomaly"] = (
#     df["cash_price_sum"] < df["cash_price_sum"].quantile(0.05)
# ).astype(int)
# # 2. Quantity anomaly features
# df["quantity_anomaly"] = (
#     df["Nbr_of_prod_purchas_sum"] > df["Nbr_of_prod_purchas_sum"].quantile(0.95)
# ).astype(int)
# ```
#
# This code relies on the computation of quantiles. The way it was programmed presented
# a data leakage issue because the quantiles were computed on the full dataset.
#
# So here, we would not to create a scikit-learn transformer to compute the quantiles
# on the training set and apply the transformation in a specific `transform` method to
# be able to apply the transformation on the test set.
#
# Hopefully, the `KBinsDiscretizer` from scikit-learn would provide a very similar
# transformation and we only need to select the columns and apply this transformer.

# %%
import warnings
from sklearn.preprocessing import KBinsDiscretizer

warnings.filterwarnings(
    "ignore", message="Bins whose width are too small", category=UserWarning
)

cols_derived_anomaly = ["cash_price_sum", "Nbr_of_prod_purchas_sum"]
discretizer = KBinsDiscretizer(
    n_bins=10,
    encode="onehot-dense",
    strategy="quantile",
    quantile_method="averaged_inverted_cdf",
).set_output(transform="pandas")

aggregated_features_with_anomaly = aggregated_features.skb.apply(
    skrub.ApplyToCols(discretizer, cols=cols_derived_anomaly, keep_original=True)
)
aggregated_features_with_anomaly

# %% [markdown]
#
# Then, we can apply the stateless operations as a normal user custom function.


# %%
@skrub.deferred
def add_domain_specific_features(df):
    # 3. Diversity features
    df["item_diversity"] = df["item_nunique"] / (df["Nbr_of_prod_purchas_sum"] + 1e-8)
    df["make_diversity"] = df["make_nunique"] / (df["Nbr_of_prod_purchas_sum"] + 1e-8)

    # 4. Price consistency features
    df["price_consistency"] = 1 / (df["cash_price_std"] + 1e-8)

    return df


# %% [markdown]
#
# And we need to concatenate the two tables to have the final features.

# %%
engineered_features = add_domain_specific_features(aggregated_features_with_anomaly)
engineered_features

# %% [markdown]
#
# Then, we can apply the predictive model to the final features and we will observe a
# preview of a fit/predict on the subsample of the data.

# %%
predictive_model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", HistGradientBoostingClassifier(random_state=42, max_iter=100)),
    ]
)
predictions = engineered_features.skb.apply(predictive_model, y=target)
predictions

# %%
predictions.skb.full_report()

# %% [markdown]
#
# Now the question is how can I extract the DataOps plan and potentially reuse it later.
#
# It is where we introduced the concept of learner which is the resulting predictive
# pipeline which will expose the usual `fit` / `predict` scikit-learn API with a twist.
# To get a fitted learner, one needs to call the `skb.make_learner` method.

# %%
learner = predictions.skb.make_learner(fitted=True)
learner

# %% [markdown]
#
# This learner can be serialized and deserialized.

# %%
import joblib

joblib.dump(learner, "../output/learner.joblib")

# %%
deserialized_learner = joblib.load("../output/learner.joblib")
deserialized_learner

# %% [markdown]
#
# Since `skrub` uses symbolic variables, the `fit` and `predict` methods are extended
# to accept a dictionary of dataframes or series. Let's compute the score on the
# training set

# %%
y_proba = deserialized_learner.predict_proba(
    {"baskets": baskets_experiment, "products": products_experiment}
)

# %%
roc_auc_score(baskets_experiment["fraud_flag"], y_proba[:, 1])

# %% [markdown]
#
# And on the production set.

# %%
y_proba = deserialized_learner.predict_proba(
    {"baskets": baskets_production, "products": products_production}
)

# %%
roc_auc_score(baskets_production["fraud_flag"], y_proba[:, 1])

# %% [markdown]
#
# ## Advanced usages of the DataOps
#
# Two usages that we did not present here yet are:
#
# - how to cross-validate a DataOps plan
# - how to tune hyperparameters of the DataOps plan
#
# The former is quite straightforward as a DataOps expose a `.skb.cross_validate`
# method. Restarting from the `predictions` variable, we can cross-validate the
# DataOps plan.

# %%
predictions

# %%
predictions.skb.cross_validate(scoring="roc_auc", cv=5, n_jobs=-1)

# %% [markdown]
#
# When it comes hyperparameters tuning, `skrub` introduces some functions that can be
# used where you want to tweak the hyperparameters. Those functions are:
#
# - `choose_from`: select from the given list of options
# - `choose_int`: select an integer within a range
# - `choose_float`: select a float within a range
# - `choose_bool`: select a bool
# - `optional`: chooses whether to execute the given operation
#
# Let's give an example where we will tune the estimators as the LLM suggested.

# %%
models

# %%
models = {name: skrub.tabular_pipeline(model) for name, model in models.items()}

# %%
predictions = engineered_features.skb.apply(
    skrub.choose_from(models, name="predictor"), y=target
)
predictions

# %%
search = predictions.skb.make_grid_search(
    fitted=True, scoring="roc_auc", cv=2, n_jobs=-1
)
search.results_

# %%
search.plot_results()

# %% [markdown]
#
# And what if we would like to perform a nested cross-validation?

# %%
nested_search = skrub.cross_validate(
    predictions.skb.make_grid_search(scoring="roc_auc", cv=2),
    environment={"baskets": baskets_experiment, "products": products_experiment},
    scoring="roc_auc",
    cv=2,
    n_jobs=-1,
    return_learner=True,
)

# %%
nested_search

# %%
for learner in nested_search["learner"]:
    print(learner.results_)
    print()

# %%
