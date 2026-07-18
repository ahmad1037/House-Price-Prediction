from pydantic import BaseModel
class PredictionResponse(BaseModel):
    predicted_price: float
class HouseFeatures(BaseModel):

    OverallQual: int

    GrLivArea: float

    GarageCars: int

    TotalBsmtSF: float

    YearBuilt: int

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
    )

    TotalBsmtSF: float = Field(
        ge=0,
    )

    YearBuilt: int = Field(
        ge=1800,
        le=2030,
    )