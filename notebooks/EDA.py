
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/train.csv")

print("=" * 60)
print("REAL ESTATE DATA ANALYSIS")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Info:")
df.info()


# ============================================================
# 3. DUPLICATE VALUES
# ============================================================

print("\n" + "=" * 60)
print("DUPLICATE CHECK")
print("=" * 60)

duplicate_count = df.duplicated().sum()

print("Number of duplicate rows:", duplicate_count)


# ============================================================
# 4. MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0].sort_values(
    ascending=False
)

print(missing_values)

print("\nTotal columns with missing values:")
print(len(missing_values))


# ============================================================
# 5. STATISTICAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STATISTICAL SUMMARY")
print("=" * 60)

print(df.describe())


# ============================================================
# 6. SALE PRICE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SALE PRICE STATISTICS")
print("=" * 60)

print(df["SalePrice"].describe())

print("\nAverage Sale Price:")
print(df["SalePrice"].mean())

print("\nMedian Sale Price:")
print(df["SalePrice"].median())

print("\nMinimum Sale Price:")
print(df["SalePrice"].min())

print("\nMaximum Sale Price:")
print(df["SalePrice"].max())


# ============================================================
# 7. NUMERICAL FEATURES
# ============================================================

numeric_df = df.select_dtypes(include=np.number)

print("\n" + "=" * 60)
print("NUMERICAL FEATURES")
print("=" * 60)

print("Number of numerical columns:", len(numeric_df.columns))


# ============================================================
# 8. CORRELATION WITH SALE PRICE
# ============================================================

correlation = numeric_df.corr()["SalePrice"].sort_values(
    ascending=False
)

print("\n" + "=" * 60)
print("TOP FEATURES CORRELATED WITH SALE PRICE")
print("=" * 60)

print(correlation.head(11))


# ============================================================
# 9. CREATE PLOTS FOLDER
# ============================================================

plot_folder = "notebooks/plots"

os.makedirs(plot_folder, exist_ok=True)


# ============================================================
# 10. SALE PRICE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df["SalePrice"],
    kde=True
)

plt.title("House Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/01_price_distribution.png",
    dpi=300
)

plt.close()


# ============================================================
# 11. SALE PRICE BOX PLOT
# ============================================================

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=df["SalePrice"]
)

plt.title("Sale Price Box Plot")
plt.xlabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/02_price_boxplot.png",
    dpi=300
)

plt.close()


# ============================================================
# 12. LIVING AREA VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="GrLivArea",
    y="SalePrice"
)

plt.title("Living Area vs Sale Price")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/03_living_area_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 13. OVERALL QUALITY VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="OverallQual",
    y="SalePrice"
)

plt.title("Overall Quality vs Sale Price")
plt.xlabel("Overall Quality")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/04_quality_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 14. YEAR BUILT VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="YearBuilt",
    y="SalePrice"
)

plt.title("Year Built vs Sale Price")
plt.xlabel("Year Built")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/05_year_built_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 15. GARAGE CARS VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="GarageCars",
    y="SalePrice"
)

plt.title("Garage Capacity vs Sale Price")
plt.xlabel("Number of Garage Cars")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/06_garage_cars_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 16. BEDROOMS VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="BedroomAbvGr",
    y="SalePrice"
)

plt.title("Bedrooms vs Sale Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/07_bedrooms_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 17. BATHROOMS VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="FullBath",
    y="SalePrice"
)

plt.title("Bathrooms vs Sale Price")
plt.xlabel("Number of Full Bathrooms")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/08_bathrooms_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 18. TOTAL ROOMS VS SALE PRICE
# ============================================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="TotRmsAbvGrd",
    y="SalePrice"
)

plt.title("Total Rooms vs Sale Price")
plt.xlabel("Total Rooms Above Ground")
plt.ylabel("Sale Price")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/09_rooms_vs_price.png",
    dpi=300
)

plt.close()


# ============================================================
# 19. TOP CORRELATIONS HEATMAP
# ============================================================

top_features = correlation.head(11).index

plt.figure(figsize=(12, 8))

sns.heatmap(
    df[top_features].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Top Features Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/10_correlation_heatmap.png",
    dpi=300
)

plt.close()


# ============================================================
# 20. TOP 10 CORRELATED FEATURES BAR CHART
# ============================================================

top_correlations = correlation.drop("SalePrice").head(10)

plt.figure(figsize=(10, 6))

top_correlations.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Features Correlated with Sale Price")
plt.xlabel("Correlation")
plt.ylabel("Feature")

plt.tight_layout()

plt.savefig(
    f"{plot_folder}/11_top_correlations.png",
    dpi=300
)

plt.close()


# ============================================================
# 21. FINAL EDA SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nDataset Shape:", df.shape)

print("\nTotal Numerical Features:", len(numeric_df.columns))

print(
    "\nColumns with Missing Values:",
    len(missing_values)
)

print(
    "\nDuplicate Rows:",
    duplicate_count
)

print(
    "\nAverage Property Price:",
    round(df["SalePrice"].mean(), 2)
)

print(
    "\nMedian Property Price:",
    round(df["SalePrice"].median(), 2)
)

print("\nTop 10 Price Correlations:")
print(top_correlations)

print("\nGraphs saved in:")
print(plot_folder)

print("\nEDA process finished!")