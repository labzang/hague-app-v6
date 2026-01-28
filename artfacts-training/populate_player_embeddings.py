from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm
# 위에서 만든 모델들 import
from your_app.models import PlayerEmbedding, TeamEmbedding, ...  # 실제 경로

engine = create_engine("postgresql+psycopg://user:pass@host/dbname")
SessionLocal = sessionmaker(bind=engine)

embedder = KoElectraEmbeddings("./koelectra_orchestrator_finetuned")

def populate_player_embeddings():
    with SessionLocal() as db:
        players = db.query(Player).all()  # Player 모델 가정
        for player in tqdm(players):
            # content 조합 (당신이 원하는 대로)
            content = f"{player.player_name}, {player.e_player_name}, {player.position}, {player.nation}, 등번호 {player.back_no}, {player.nickname}"
            if not content.strip():
                continue

            embedding = embedder.embed_query(content)

            existing = db.query(PlayerEmbedding).filter_by(player_id=player.id).first()
            if existing:
                # 업데이트하거나 스킵
                continue

            emb_record = PlayerEmbedding(
                player_id=player.id,
                content=content,
                embedding=np.array(embedding)
            )
            db.add(emb_record)
        db.commit()

# teams, schedules, stadiums도 동일 패턴으로
