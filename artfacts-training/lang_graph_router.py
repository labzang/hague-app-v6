from langgraph.graph import StateGraph
from langchain_postgres import PGVector  # 또는 langchain_community.vectorstores.pgvector

# 분류기 로드
classifier_tokenizer = AutoTokenizer.from_pretrained("./koelectra_orchestrator_finetuned")
classifier_model = AutoModelForSequenceClassification.from_pretrained("./koelectra_orchestrator_finetuned")

def classify_query(query: str) -> int:
    inputs = classifier_tokenizer(query, return_tensors="pt", truncation=True)
    outputs = classifier_model(**inputs)
    return torch.argmax(outputs.logits, dim=1).item()  # 0=시맨틱, 1=규칙

# 벡터스토어 (각 엔티티별로 따로 만들기 추천)
player_vectorstore = PGVector(
    connection="postgresql+psycopg://...",
    collection_name="players",
    embeddings=KoElectraEmbeddings(),
    use_jsonb=True,
)

# LangGraph 노드 예시
def router(state):
    query = state["query"]
    cls = classify_query(query)
    if cls == 0:  # 시맨틱 → 벡터 검색
        docs = player_vectorstore.similarity_search(query, k=5)  # + 다른 vectorstore도
        state["retrieved"] = docs
        return "rag_node"
    else:
        return "rule_node"
