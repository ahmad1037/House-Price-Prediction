from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    OverallQual: int = Field(
        ge=1,
        le=10,
        description="Overall material and finish quality",
    )

    GrLivArea: float = Field(
        gt=0,
        description="Above-ground living area in square feet",
    )

    GarageCars: int = Field(
        ge=0,
        le=6,
        description="Number of cars the garage can hold",
    )

    TotalBsmtSF: float = Field(
        ge=0,
        description="Total basement area in square feet",
    )

    YearBuilt: int = Field(
        ge=1800,
        le=2030,
        description="Original construction year",
    )


class PredictionResponse(BaseModel):
    predicted_price: float