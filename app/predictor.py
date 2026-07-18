from src.model_io import load_object
model = load_object(
    "best_model.joblib"
)

preprocessor = load_object(
    "preprocessor.joblib"
)

import pandas as pd

def predict(
    input_data: dict,
):
    """
    Predict house price from raw input.
    """

    df = pd.DataFrame(
        [input_data]
    )

    processed = preprocessor.transform(df)

    prediction = model.predict(
        processed
    )

    return float(
        prediction[0]
    )

import pandas as pd

def predict(
    input_data: dict,
):
    """
    Predict house price from raw input.
    """

    df = pd.DataFrame(
        [input_data]
    )

    processed = preprocessor.transform(df)

    prediction = model.predict(
        processed
    )

    return float(
        prediction[0]
    )

sample = {
    "Order": 3000,
    "PID": 5312345678,
    "MS SubClass": 20,
    "MS Zoning": "RL",
    "Lot Frontage": 75,
    "Lot Area": 9500,
    "Street": "Pave",
    "Alley": "NA",
    "Lot Shape": "Reg",
    "Land Contour": "Lvl",
    "Utilities": "AllPub",
    "Lot Config": "Inside",
    "Land Slope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition 1": "Norm",
    "Condition 2": "Norm",
    "Bldg Type": "1Fam",
    "House Style": "1Story",
    "Overall Qual": 7,
    "Overall Cond": 6,
    "Year Built": 2005,
    "Year Remod/Add": 2015,
    "Roof Style": "Gable",
    "Roof Matl": "CompShg",
    "Exterior 1st": "VinylSd",
    "Exterior 2nd": "VinylSd",
    "Mas Vnr Type": "Stone",
    "Mas Vnr Area": 150,
    "Exter Qual": "Gd",
    "Exter Cond": "TA",
    "Foundation": "PConc",
    "Bsmt Qual": "Gd",
    "Bsmt Cond": "TA",
    "Bsmt Exposure": "Av",
    "BsmtFin Type 1": "GLQ",
    "BsmtFin SF 1": 650,
    "BsmtFin Type 2": "Unf",
    "BsmtFin SF 2": 0,
    "Bsmt Unf SF": 450,
    "Total Bsmt SF": 1100,
    "Heating": "GasA",
    "Heating QC": "Ex",
    "Central Air": "Y",
    "Electrical": "SBrkr",
    "1st Flr SF": 1100,
    "2nd Flr SF": 600,
    "Low Qual Fin SF": 0,
    "Gr Liv Area": 1700,
    "Bsmt Full Bath": 1,
    "Bsmt Half Bath": 0,
    "Full Bath": 2,
    "Half Bath": 1,
    "Bedroom AbvGr": 3,
    "Kitchen AbvGr": 1,
    "Kitchen Qual": "Gd",
    "TotRms AbvGrd": 7,
    "Functional": "Typ",
    "Fireplaces": 1,
    "Fireplace Qu": "Gd",
    "Garage Type": "Attchd",
    "Garage Yr Blt": 2005,
    "Garage Finish": "Fin",
    "Garage Cars": 2,
    "Garage Area": 500,
    "Garage Qual": "TA",
    "Garage Cond": "TA",
    "Paved Drive": "Y",
    "Wood Deck SF": 120,
    "Open Porch SF": 40,
    "Enclosed Porch": 0,
    "3Ssn Porch": 0,
    "Screen Porch": 0,
    "Pool Area": 0,
    "Pool QC": "NA",
    "Fence": "NA",
    "Misc Feature": "NA",
    "Misc Val": 0,
    "Mo Sold": 6,
    "Yr Sold": 2010,
    "Sale Type": "WD ",
    "Sale Condition": "Normal",
}

price = predict(sample)
print(f"Predicted Price: ${price:,.2f}")