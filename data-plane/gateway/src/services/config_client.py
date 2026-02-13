"""
Config client for polling control plane.

This demonstrates the service discovery and config sync pattern commonly
used in distributed systems.
"""

import httpx
import asyncio
from typing import Optional, Dict


class ConfigClient:
    """
    Client for polling routing configuration from control plane.

    Implements:
    - Periodic polling (every 30s)
    - Graceful error handling (keeps using old config if poll fails)
    - Callback-based updates (notifies router when config changes)
    """

    def __init__(
        self,
        config_service_url: str,
        poll_interval: int = 30,
        on_config_update: Optional[callable] = None
    ):
        """
        Initialize config client.

        Args:
            config_service_url: Config service URL
            poll_interval: Polling interval in seconds
            on_config_update: Callback function called when config changes
        """
        self.config_service_url = config_service_url
        self.poll_interval = poll_interval
        self.on_config_update = on_config_update

        self._current_version = 0
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def fetch_config(self) -> Optional[Dict]:
        """
        Fetch routing config from control plane.

        Returns:
            Config dict if successful, None otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self.config_service_url}/api/v1/config/routing"
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    print(f"Failed to fetch config: HTTP {response.status_code}")
                    return None

        except httpx.TimeoutException:
            print("Config fetch timed out")
            return None
        except httpx.ConnectError:
            print(f"Cannot connect to config service at {self.config_service_url}")
            return None
        except Exception as e:
            print(f"Error fetching config: {e}")
            return None

    async def poll_loop(self):
        """
        Main polling loop.

        Runs in background, periodically fetching config and notifying
        on changes.
        """
        print(f"Starting config polling loop (interval: {self.poll_interval}s)")

        while self._running:
            try:
                # Fetch config
                config_data = await self.fetch_config()

                if config_data:
                    # Check if version changed
                    new_version = config_data.get("version", 0)
                    if new_version > self._current_version:
                        self._current_version = new_version

                        # Extract normalized weights
                        weights = config_data.get("normalized_weights", {})

                        print(f"✓ Config updated to version {new_version}")
                        print(f"  Routing weights: {weights}")

                        # Notify callback
                        if self.on_config_update and weights:
                            self.on_config_update(weights)

            except Exception as e:
                print(f"Error in polling loop: {e}")

            # Wait before next poll
            await asyncio.sleep(self.poll_interval)

    def start(self):
        """Start the polling loop."""
        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self.poll_loop())
            print("Config polling started")

    def stop(self):
        """Stop the polling loop."""
        if self._running:
            self._running = False
            if self._task:
                self._task.cancel()
            print("Config polling stopped")
