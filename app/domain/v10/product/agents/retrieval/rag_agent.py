"""
RAG 에이전트
기존 rag_service.py를 에이전트로 변환
"""

from typing import Dict, Any
from app.agents.base_agent import BaseAgent


class RAGAgent(BaseAgent):
    """RAG (Retrieval-Augmented Generation) 에이전트"""

    def __init__(self):
        super().__init__(
            name="rag_agent",
            instruction="""You are a RAG (Retrieval-Augmented Generation) agent.
            Your role is to:
            1. Retrieve relevant documents for user queries
            2. Generate contextual responses using retrieved information
            3. Combine search results with language model capabilities
            4. Provide accurate, source-backed answers
            """,
            server_names=["filesystem"],  # 문서 접근용
            metadata={
                "approach": "Retrieval-Augmented Generation",
                "components": ["Vector Search", "LLM Generation"],
                "languages": ["Korean", "English"]
            }
        )

    async def execute(self, task: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """RAG 파이프라인 실행"""
        question = context.get("question", task)
        k = context.get("k", 3)  # 검색할 문서 수

        # 1. 문서 검색 단계
        search_results = await self._retrieve_documents(question, k)

        # 2. 응답 생성 단계
        generated_answer = await self._generate_answer(question, search_results)

        # 3. 결과 구성
        return {
            "question": question,
            "answer": generated_answer,
            "sources": search_results,
            "retrieved_count": len(search_results),
            "rag_pipeline": {
                "retrieval": "completed",
                "generation": "completed",
                "total_steps": 2
            }
        }

    async def _retrieve_documents(self, question: str, k: int) -> List[Dict[str, Any]]:
        """문서 검색 단계"""
        # TODO: 실제 벡터 검색 통합
        # VectorSearchAgent를 호출하거나 직접 vectorstore 사용

        # 현재는 모킹 구현
        mock_documents = [
            {
                "content": f"검색된 문서 {i+1}: {question}에 대한 상세한 정보를 포함하고 있습니다.",
                "metadata": {
                    "source": f"knowledge_base_{i+1}.md",
                    "relevance_score": 0.85 - (i * 0.1),
                    "section": f"Section {i+1}"
                }
            }
            for i in range(min(k, 3))
        ]

        return mock_documents

    async def _generate_answer(self, question: str, documents: List[Dict[str, Any]]) -> str:
        """답변 생성 단계"""
        # TODO: 실제 LLM 통합 (EXAONE, OpenAI 등)

        # 검색된 문서들을 컨텍스트로 구성
        context = "\n".join([doc["content"] for doc in documents])

        # 현재는 간단한 템플릿 기반 응답
        if not documents:
            return f"'{question}'에 대한 관련 문서를 찾지 못했습니다. 더 구체적인 질문을 해주시면 도움이 될 것 같습니다."

        return f"""'{question}'에 대한 답변입니다:

검색된 {len(documents)}개의 문서를 바탕으로 다음과 같이 답변드립니다:

{context[:200]}...

이 정보가 도움이 되셨나요? 더 자세한 내용이 필요하시면 구체적으로 질문해주세요."""

    def get_rag_stats(self) -> Dict[str, Any]:
        """RAG 통계 조회"""
        return {
            "total_queries": self.execution_count,
            "last_query": self.last_execution.isoformat() if self.last_execution else None,
            "pipeline_components": [
                "Document Retrieval",
                "Context Assembly",
                "Answer Generation",
                "Source Attribution"
            ],
            "supported_formats": ["Text", "Markdown", "JSON"]
        }
