"""Soccer 도메인 Repository 모듈."""

from app.domain.v10.soccer.hub.repositories.player_repository import PlayerRepository
from app.domain.v10.soccer.hub.repositories.schedule_repository import ScheduleRepository
from app.domain.v10.soccer.hub.repositories.stadium_repository import StadiumRepository
from app.domain.v10.soccer.hub.repositories.team_repository import TeamRepository

__all__ = [
    "PlayerRepository",
    "ScheduleRepository",
    "StadiumRepository",
    "TeamRepository",
]
