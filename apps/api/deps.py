from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from api.config import Settings, get_settings
from api.db.session import get_db

SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]
