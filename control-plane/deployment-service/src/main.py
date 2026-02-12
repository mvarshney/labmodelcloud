# File: /ml-model-serving-platform/ml-model-serving-platform/control-plane/deployment-service/src/main.py

from fastapi import FastAPI
from api.routes import router as deployment_router

app = FastAPI(title="Deployment Service")

app.include_router(deployment_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Deployment Service!"}