import pytest

from scan.config import settings


@pytest.fixture(scope="session", autouse=True)
def patch_settings():
    settings.OPENAI_API_KEY = "my_key"
    settings.SERPER_API_KEY = "search_key"
    yield
