import pandas as pd


def load_and_preprocess(path="data/train.csv"):
    df = pd.read_csv(path)

    # Target
    y = df["SalePrice"]

    # Remove target and ID
    X = df.drop(columns=["SalePrice", "Id"])

    # Columns where missing means the feature does not exist
    none_columns = [
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "MasVnrType",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
    ]

    for col in none_columns:
        if col in X.columns:
            X[col] = X[col].fillna("None")

    # Numerical columns where missing values can be replaced by median
    numerical_columns = [
        "LotFrontage",
        "MasVnrArea",
        "GarageYrBlt",
    ]

    for col in numerical_columns:
        if col in X.columns:
            X[col] = X[col].fillna(X[col].median())

    # Electrical has only one missing value
    if "Electrical" in X.columns:
        X["Electrical"] = X["Electrical"].fillna(X["Electrical"].mode()[0])

    # Any remaining numerical missing values
    for col in X.select_dtypes(include=["int64", "float64"]).columns:
        X[col] = X[col].fillna(X[col].median())

    # Any remaining categorical missing values
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].fillna("None")

    return X, y


if __name__ == "__main__":
    X, y = load_and_preprocess()

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    print("Remaining missing values:", X.isnull().sum().sum())