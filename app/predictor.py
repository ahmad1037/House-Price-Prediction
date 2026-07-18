from turtle import pd
import pandas as pd
from src.model_io import load_object
model = load_object(
    "best_model.joblib"
)

preprocessor = load_object(
    "preprocessor.joblib"
)

DEFAULT_INPUT = {
    "Order": 0,
    "PID": 0,
    "MS SubClass": 20,
    "MS Zoning": "RL",
    "Lot Frontage": 70,
    "Lot Area": 9000,
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
    "Overall Qual": 5,
    "Overall Cond": 5,
    "Year Built": 1975,
    "Year Remod/Add": 1975,
    "Roof Style": "Gable",
    "Roof Matl": "CompShg",
    "Exterior 1st": "VinylSd",
    "Exterior 2nd": "VinylSd",
    "Mas Vnr Type": "None",
    "Mas Vnr Area": 0,
    "Exter Qual": "TA",
    "Exter Cond": "TA",
    "Foundation": "CBlock",
    "Bsmt Qual": "TA",
    "Bsmt Cond": "TA",
    "Bsmt Exposure": "No",
    "BsmtFin Type 1": "Unf",
    "BsmtFin SF 1": 0,
    "BsmtFin Type 2": "Unf",
    "BsmtFin SF 2": 0,
    "Bsmt Unf SF": 800,
    "Total Bsmt SF": 800,
    "Heating": "GasA",
    "Heating QC": "TA",
    "Central Air": "Y",
    "Electrical": "SBrkr",
    "1st Flr SF": 1000,
    "2nd Flr SF": 0,
    "Low Qual Fin SF": 0,
    "Gr Liv Area": 1000,
    "Bsmt Full Bath": 0,
    "Bsmt Half Bath": 0,
    "Full Bath": 1,
    "Half Bath": 0,
    "Bedroom AbvGr": 3,
    "Kitchen AbvGr": 1,
    "Kitchen Qual": "TA",
    "TotRms AbvGrd": 6,
    "Functional": "Typ",
    "Fireplaces": 0,
    "Fireplace Qu": "NA",
    "Garage Type": "Attchd",
    "Garage Yr Blt": 1975,
    "Garage Finish": "Unf",
    "Garage Cars": 2,
    "Garage Area": 480,
    "Garage Qual": "TA",
    "Garage Cond": "TA",
    "Paved Drive": "Y",
    "Wood Deck SF": 0,
    "Open Porch SF": 0,
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

def predict(input_data: dict):
    data = DEFAULT_INPUT.copy()
    data.update(input_data)

    df = pd.DataFrame([data])

    processed = preprocessor.transform(df)

    prediction = model.predict(processed)

    return float(prediction[0])
