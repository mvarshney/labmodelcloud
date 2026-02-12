from fastapi import APIRouter

router = APIRouter()

@router.get("/config")
async def get_config():
    return {"message": "Retrieve routing configuration"}

@router.post("/config")
async def update_config(config: dict):
    return {"message": "Update routing configuration", "config": config}