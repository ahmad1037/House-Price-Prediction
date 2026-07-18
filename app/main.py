from fastapi import FastAPI

app = FastAPI(
    title="House Price Prediction API",
    version="1.0.0",
)
@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is running."
    }