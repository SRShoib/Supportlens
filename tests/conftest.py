import random

import pytest


@pytest.fixture(autouse=True)
def _fixed_seed() -> None:
    random.seed(42)
