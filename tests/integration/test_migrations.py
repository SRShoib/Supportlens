import pytest

from alembic import command
from alembic.config import Config

pytestmark = pytest.mark.integration


def _alembic_config(database_url: str) -> Config:
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def test_upgrade_downgrade_upgrade_cycle(database_url: str) -> None:
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    command.downgrade(config, "base")
    command.upgrade(config, "head")


def test_no_autogenerate_drift_against_models(database_url: str) -> None:
    """alembic check fails (raises) if the ORM models have diverged from what
    the committed migration actually creates."""
    config = _alembic_config(database_url)
    command.upgrade(config, "head")
    command.check(config)
