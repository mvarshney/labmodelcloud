from fastapi import FastAPI
from .api.routes import router as api_router

app = FastAPI(title="Config Service")

app.include_router(api_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the Config Service!"}