# %% [markdown]
#
# # Data exploration

# %%
from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()

# %%
dataset.employee_salaries.head()

# %% [markdown]
#
# Let's ask Claude to explore this dataframe for us:
#
# Prompt:
#
# I would like to explore the `dataset.employee_salaries` dataframe to get some
# insights about what data am I facing?
#
# Could you help me, please?


# %%
# Basic information about the dataset
print("Dataset shape:", dataset.employee_salaries.shape)
print("\nColumn names and data types:")
print(dataset.employee_salaries.dtypes)

# %%
# Dataset information
print("Dataset information:")
print(
    f"Dataset name: {dataset.name if hasattr(dataset, 'name') else 'Employee Salaries'}"
)
print(f"Number of features: {len(dataset.employee_salaries.columns)}")
print(f"Number of samples: {len(dataset.employee_salaries)}")

# %%
# Basic statistics for numerical columns
print("Numerical columns statistics:")
print(dataset.employee_salaries.describe())

# %%
# Check for missing values
print("Missing values per column:")
print(dataset.employee_salaries.isnull().sum())

# %%
# Unique values in categorical columns
print("Unique values in categorical columns:")
for col in dataset.employee_salaries.select_dtypes(include=["object"]).columns:
    unique_count = dataset.employee_salaries[col].nunique()
    print(f"\n{col}: {unique_count} unique values")
    if unique_count <= 20:  # Show values if not too many
        print(f"Values: {dataset.employee_salaries[col].unique()}")
    else:
        print(f"Sample values: {dataset.employee_salaries[col].unique()[:10]}")

# %%
# Check the target variable distribution
if "salary" in dataset.employee_salaries.columns:
    print("Salary distribution:")
    print(dataset.employee_salaries["salary"].describe())
    print(
        f"\nSalary range: {dataset.employee_salaries['salary'].min():.2f} - "
        f"{dataset.employee_salaries['salary'].max():.2f}"
    )

# %%
# Sample of the data
print("First 10 rows of the dataset:")
print(dataset.employee_salaries.head(10))

# %%
# Check for any potential data quality issues
print("Data quality check:")
print(f"Duplicate rows: {dataset.employee_salaries.duplicated().sum()}")
print(f"Rows with all NaN: {dataset.employee_salaries.isnull().all(axis=1).sum()}")

# %% [markdown]
#
# Pretty verbose and not actual plots. Let's request some:
#
# Prompt:
#
# Please add also a couple of visual display to check the distribution of the
# different type of data.

# %%
# Import visualization libraries
import matplotlib.pyplot as plt

# %%
# Create a comprehensive visualization dashboard
fig, axes = plt.subplots(3, 3, figsize=(20, 15))
fig.suptitle(
    "Employee Salaries Dataset - Data Distribution Analysis",
    fontsize=16,
    fontweight="bold",
)

# 1. Salary distribution (histogram)
axes[0, 0].hist(
    dataset.employee_salaries["current_annual_salary"],
    bins=50,
    alpha=0.7,
    edgecolor="black",
)
axes[0, 0].set_title("Salary Distribution")
axes[0, 0].set_xlabel("Annual Salary ($)")
axes[0, 0].set_ylabel("Frequency")
axes[0, 0].ticklabel_format(style="plain", axis="x")

# 2. Salary distribution (box plot)
axes[0, 1].boxplot(dataset.employee_salaries["current_annual_salary"])
axes[0, 1].set_title("Salary Distribution (Box Plot)")
axes[0, 1].set_ylabel("Annual Salary ($)")
axes[0, 1].ticklabel_format(style="plain", axis="y")

# 3. Gender distribution
gender_counts = dataset.employee_salaries["gender"].value_counts()
axes[0, 2].pie(
    gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%", startangle=90
)
axes[0, 2].set_title("Gender Distribution")

# 4. Top 10 departments by employee count
dept_counts = dataset.employee_salaries["department_name"].value_counts().head(10)
axes[1, 0].barh(range(len(dept_counts)), dept_counts.values)
axes[1, 0].set_yticks(range(len(dept_counts)))
axes[1, 0].set_yticklabels(
    [name[:30] + "..." if len(name) > 30 else name for name in dept_counts.index]
)
axes[1, 0].set_title("Top 10 Departments by Employee Count")
axes[1, 0].set_xlabel("Number of Employees")

# 5. Assignment category distribution
assignment_counts = dataset.employee_salaries["assignment_category"].value_counts()
axes[1, 1].bar(assignment_counts.index, assignment_counts.values)
axes[1, 1].set_title("Assignment Category Distribution")
axes[1, 1].set_ylabel("Number of Employees")
axes[1, 1].tick_params(axis="x", rotation=45)

# 6. Year first hired distribution
year_counts = dataset.employee_salaries["year_first_hired"].value_counts().sort_index()
axes[1, 2].plot(
    year_counts.index, year_counts.values, marker="o", linewidth=2, markersize=4
)
axes[1, 2].set_title("Hiring Trends Over Time")
axes[1, 2].set_xlabel("Year First Hired")
axes[1, 2].set_ylabel("Number of Employees")
axes[1, 2].grid(True, alpha=0.3)

# 7. Salary by gender (box plot)
salary_by_gender = [
    dataset.employee_salaries[dataset.employee_salaries["gender"] == gender][
        "current_annual_salary"
    ].dropna()
    for gender in ["F", "M"]
]
axes[2, 0].boxplot(salary_by_gender, tick_labels=["Female", "Male"])
axes[2, 0].set_title("Salary Distribution by Gender")
axes[2, 0].set_ylabel("Annual Salary ($)")
axes[2, 0].ticklabel_format(style="plain", axis="y")

# 8. Salary by assignment category (box plot)
salary_by_category = [
    dataset.employee_salaries[dataset.employee_salaries["assignment_category"] == cat][
        "current_annual_salary"
    ]
    for cat in dataset.employee_salaries["assignment_category"].unique()
]
axes[2, 1].boxplot(
    salary_by_category,
    tick_labels=dataset.employee_salaries["assignment_category"].unique(),
)
axes[2, 1].set_title("Salary Distribution by Assignment Category")
axes[2, 1].set_ylabel("Annual Salary ($)")
axes[2, 1].ticklabel_format(style="plain", axis="y")
axes[2, 1].tick_params(axis="x", rotation=45)

# 9. Experience vs Salary scatter plot
axes[2, 2].scatter(
    dataset.employee_salaries["year_first_hired"],
    dataset.employee_salaries["current_annual_salary"],
    alpha=0.5,
    s=10,
)
axes[2, 2].set_title("Experience vs Current Salary")
axes[2, 2].set_xlabel("Year First Hired")
axes[2, 2].set_ylabel("Current Annual Salary ($)")
axes[2, 2].ticklabel_format(style="plain", axis="y")
axes[2, 2].grid(True, alpha=0.3)

# %%
# Additional detailed analysis - Top departments by average salary
print("Top 10 departments by average salary:")
dept_salary = (
    dataset.employee_salaries.groupby("department_name")["current_annual_salary"]
    .agg(["mean", "count"])
    .round(2)
)
dept_salary = dept_salary[
    dept_salary["count"] >= 10
]  # Only departments with at least 10 employees
top_depts = dept_salary.sort_values("mean", ascending=False).head(10)
print(top_depts)

# %%
# Salary statistics by gender
print("\nSalary statistics by gender:")
gender_salary_stats = dataset.employee_salaries.groupby("gender")[
    "current_annual_salary"
].describe()
print(gender_salary_stats.round(2))

# %%
# Create a correlation heatmap for numerical variables using matplotlib
plt.figure(figsize=(8, 6))
numerical_data = dataset.employee_salaries[
    ["year_first_hired", "current_annual_salary"]
]
correlation_matrix = numerical_data.corr()

# Create heatmap using matplotlib
im = plt.imshow(correlation_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(im, shrink=0.8)

# Add text annotations
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.3f}",
            ha="center",
            va="center",
            fontweight="bold",
        )

plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.title("Correlation Matrix - Numerical Variables")

# %% [markdown]
#
# Can we do better knowing that this process of exploring data is recurrent.

# %%
from skrub import TableReport

report = TableReport(dataset.employee_salaries)
report

# %%
report.write_html("../output/report.html")

# %%
