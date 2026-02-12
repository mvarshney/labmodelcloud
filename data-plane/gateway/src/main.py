from fastapi import FastAPI
from services.router import Router
from services.config_client import ConfigClient
from services.triton_client import TritonClient
from metrics.prometheus import PrometheusMetrics

app = FastAPI()
router = Router()
config_client = ConfigClient()
triton_client = TritonClient()
metrics = PrometheusMetrics()

@app.on_event("startup")
async def startup_event():
    await config_client.sync_config()

@app.post("/predict")
async def predict(request: dict):
    response = await router.route_request(request)
    return response

@app.get("/health")
async def health_check():
    return {"status": "healthy"}