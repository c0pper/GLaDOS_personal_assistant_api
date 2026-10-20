from datetime import datetime

from atomic_agents.context import BaseDynamicContextProvider


class CurrentDateProvider(BaseDynamicContextProvider):
    def get_info(self) -> str:
        # Computed on every prompt build so the date doesn't freeze at process start.
        return f"Current date in format YYYY-MM-DD: {datetime.now().strftime('%Y-%m-%d')}"
