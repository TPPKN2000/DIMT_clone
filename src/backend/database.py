"""
SQLAlchemy models + session factory for eval persistence.
"""

from datetime import datetime
from sqlalchemy import create_engine, Column, String, Float, Integer, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id            = Column(String, primary_key=True)   # doc_id
    pdf_name      = Column(String, nullable=False)
    model_ver     = Column(String, default="nllb-1.3B-lora")
    created_at    = Column(DateTime, default=datetime.utcnow)
    user_rating   = Column(Integer, nullable=True)
    overall_score = Column(Float, nullable=True)

    cases    = relationship("EvalCase", back_populates="run")
    keywords = relationship("EvalKeyword", back_populates="run")


class EvalCase(Base):
    __tablename__ = "eval_cases"

    id          = Column(String, primary_key=True)
    run_id      = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    page_num    = Column(Integer)
    block_type  = Column(String)        # text | table | title
    src_text    = Column(Text)          # tiếng Anh gốc
    tgt_text    = Column(Text)          # bản dịch tiếng Việt
    metric_name = Column(String)        # TranslationFidelity | OCRQuality | ...
    score       = Column(Float)
    reason      = Column(Text)          # LLM giải thích
    is_q4       = Column(Boolean, default=False)
    created_at  = Column(DateTime, default=datetime.utcnow)

    run = relationship("EvalRun", back_populates="cases")


class EvalKeyword(Base):
    __tablename__ = "eval_keywords"

    id       = Column(String, primary_key=True)
    run_id   = Column(String, ForeignKey("eval_runs.id"), nullable=False)
    keyword  = Column(String)
    wiki_url = Column(String)

    run = relationship("EvalRun", back_populates="keywords")


# ── Session factory ─────────────────────────────────────────
engine = create_engine(
    "sqlite:///./dimt_eval.db",
    connect_args={"check_same_thread": False},
)
SessionLocal = sessionmaker(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)