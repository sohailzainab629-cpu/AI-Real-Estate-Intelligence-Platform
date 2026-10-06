
import pandas as pd


# Load real estate dataset
df = pd.read_csv("data/train.csv")


def recommend_properties(
    budget,
    neighborhood=None,
    bedrooms=None,
    top_n=5
):
    """
    Recommend properties based on:
    - Maximum budget
    - Neighborhood
    - Number of bedrooms
    """

    recommendations = df.copy()

    # Filter by budget
    recommendations = recommendations[
        recommendations["SalePrice"] <= budget
    ]

    # Filter by neighborhood if provided
    if neighborhood and neighborhood != "Any":
        recommendations = recommendations[
            recommendations["Neighborhood"].str.lower()
            == neighborhood.lower()
        ]

    # Filter by bedrooms if provided
    if bedrooms is not None:
        recommendations = recommendations[
            recommendations["BedroomAbvGr"] >= bedrooms
        ]

    # Sort by price
    recommendations = recommendations.sort_values(
        by="SalePrice",
        ascending=False
    )

    # Select useful columns
    result = recommendations[
        [
            "Id",
            "SalePrice",
            "Neighborhood",
            "BedroomAbvGr",
            "FullBath",
            "GrLivArea",
            "OverallQual"
        ]
    ].head(top_n)

    return result.to_dict(orient="records")
