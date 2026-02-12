# config_client.py

import httpx
import asyncio

class ControlPlaneClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def get_routing_config(self):
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/config")
            response.raise_for_status()
            return response.json()

class ConfigClient:
    def __init__(self, control_plane_client: ControlPlaneClient):
        self.control_plane_client = control_plane_client

    async def fetch_config(self):
        return await self.control_plane_client.get_routing_config()