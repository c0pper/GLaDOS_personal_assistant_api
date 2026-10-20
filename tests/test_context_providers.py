from datetime import datetime

from src.agents import context_providers
from src.agents.context_providers import CurrentDateProvider


def test_date_is_computed_on_every_call(monkeypatch):
    provider = CurrentDateProvider("Current Date")

    class FakeDatetime:
        value = datetime(2026, 1, 1, 23, 59)

        @classmethod
        def now(cls):
            return cls.value

    monkeypatch.setattr(context_providers, "datetime", FakeDatetime)
    assert provider.get_info() == "Current date in format YYYY-MM-DD: 2026-01-01"

    FakeDatetime.value = datetime(2026, 1, 2, 0, 1)
    assert provider.get_info() == "Current date in format YYYY-MM-DD: 2026-01-02"
