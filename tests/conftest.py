import os
from unittest.mock import patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def patch_env():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "my_key"}, clear=True):
        yield
