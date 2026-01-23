"""
검색 서비스 오케스트레이터
벡터 검색 관련 비즈니스 로직 및 서비스 함수들
"""

from typing import List, Tuple
from langchain_core.documents import Document

from app.core.vectorstore import get_vectorstore, VectorStoreType
from app.schemas import SearchRequest, SearchResponse, DocumentResponse


class SearchOrchestrator:
    """검색 오케스트레이터 클래스"""

    def __init__(self):
        self.vectorstore = None

    def get_vectorstore(self) -> VectorStoreType:
        """벡터스토어 인스턴스 가져오기 (지연 초기화)"""
        if self.vectorstore is None:
            self.vectorstore = get_vectorstore()
        return self.vectorstore

    async def perform_vector_search(
        self,
        query: str,
        k: int = 5
    ) -> Tuple[List[Document], List[float]]:
        """
        벡터 유사도 검색 수행

        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수

        Returns:
            Tuple[List[Document], List[float]]: (문서 목록, 점수 목록)
        """
        vectorstore = self.get_vectorstore()
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

        documents = [doc for doc, score in docs_with_scores]
        scores = [float(score) for doc, score in docs_with_scores]

        return documents, scores

    async def search_and_format(
        self,
        request: SearchRequest
    ) -> SearchResponse:
        """
        검색 수행 후 응답 형식으로 변환

        Args:
            request: 검색 요청

        Returns:
            SearchResponse: 포맷된 검색 응답
        """
        documents, scores = await self.perform_vector_search(
            request.query, request.k
        )

        # 응답 모델로 변환
        formatted_documents = [
            DocumentResponse(
                content=doc.page_content,
                metadata=doc.metadata,
                score=score,
            )
            for doc, score in zip(documents, scores)
        ]

        return SearchResponse(
            query=request.query,
            documents=formatted_documents,
            count=len(formatted_documents),
        )

    async def get_service_health(self) -> dict:
        """검색 서비스 상태 확인"""
        try:
            vectorstore = self.get_vectorstore()
            # 간단한 테스트 검색으로 상태 확인
            test_docs = vectorstore.similarity_search("test", k=1)
            return {
                "status": "healthy",
                "service": "vector_search",
                "vectorstore_available": True,
                "test_search_successful": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "vector_search",
                "vectorstore_available": False,
                "error": str(e)
            }


# 싱글톤 인스턴스
_search_orchestrator: SearchOrchestrator = None

def get_search_orchestrator() -> SearchOrchestrator:
    """검색 오케스트레이터 싱글톤 인스턴스 가져오기"""
    global _search_orchestrator
    if _search_orchestrator is None:
        _search_orchestrator = SearchOrchestrator()
    return _search_orchestrator
