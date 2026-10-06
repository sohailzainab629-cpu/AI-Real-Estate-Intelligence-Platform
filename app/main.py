from typing import Optional

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.ai_assistant import ask_ai
from app.model import predict_price
from app.recommender import recommend_properties


app = FastAPI(
    title="AI Real Estate Intelligence Platform",
    version="1.0.0"
)


# =========================================================
# STATIC FILES AND TEMPLATES
# =========================================================

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# =========================================================
# REQUEST MODELS
# =========================================================

class PredictionRequest(BaseModel):
    OverallQual: int
    GrLivArea: float
    GarageCars: float
    GarageArea: float
    TotalBsmtSF: float
    firstFlrSF: float
    FullBath: float
    TotRmsAbvGrd: int
    YearBuilt: int
    YearRemodAdd: int
    BedroomAbvGr: int
    Neighborhood: str
    HouseStyle: str
    KitchenQual: str


class RecommendationRequest(BaseModel):
    budget: float
    neighborhood: Optional[str] = "Any"
    bedrooms: Optional[int] = None
    top_n: int = 5


class ChatRequest(BaseModel):
    question: str


# =========================================================
# FRONTEND HOME PAGE
# =========================================================

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


# =========================================================
# PRICE PREDICTION
# =========================================================

@app.post("/predict")
def predict(request: PredictionRequest):

    data = {
        "OverallQual": request.OverallQual,
        "GrLivArea": request.GrLivArea,
        "GarageCars": request.GarageCars,
        "GarageArea": request.GarageArea,
        "TotalBsmtSF": request.TotalBsmtSF,
        "1stFlrSF": request.firstFlrSF,
        "FullBath": request.FullBath,
        "TotRmsAbvGrd": request.TotRmsAbvGrd,
        "YearBuilt": request.YearBuilt,
        "YearRemodAdd": request.YearRemodAdd,
        "BedroomAbvGr": request.BedroomAbvGr,
        "Neighborhood": request.Neighborhood,
        "HouseStyle": request.HouseStyle,
        "KitchenQual": request.KitchenQual
    }

    predicted_price = predict_price(data)

    return {
        "predicted_price": round(predicted_price, 2),
        "currency": "USD"
    }


# =========================================================
# PROPERTY RECOMMENDATION
# =========================================================

@app.post("/recommend")
def recommend(request: RecommendationRequest):

    results = recommend_properties(
        budget=request.budget,
        neighborhood=request.neighborhood,
        bedrooms=request.bedrooms,
        top_n=request.top_n
    )

    return {
        "count": len(results),
        "recommendations": results
    }


# =========================================================
# AI + RAG CHATBOT
# =========================================================

@app.post("/chat")
def chat(request: ChatRequest):

    answer = ask_ai(request.question)

    return {
        "question": request.question,
        "answer": answer
    }