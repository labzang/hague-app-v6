"""축구 도메인 라우팅 모듈.

질문 분류 및 라우팅 관련 기능을 제공합니다.
"""
from app.domain.v10.soccer.hub.routing.question_classifier import (
    QuestionClassifier,
    DomainType
)

__all__ = ["QuestionClassifier", "DomainType"]

