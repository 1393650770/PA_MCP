# [AI:BEGIN]
# PA_MCP - Info Layer: News Aggregation & Sentiment Analysis
# Aggregates financial news from multiple Chinese sources.
# Sentiment analysis using bardsai/finance-sentiment-zh-base (BERT-based, 0.1B params).
# [AI:END]

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


# ---- Data Classes ----

@dataclass
class NewsItem:
    """Unified news item from any source."""

    title: str
    content: str = ""
    source: str = ""
    url: str = ""
    pub_time: str = ""
    symbols: list[str] = field(default_factory=list)  # Related stock codes
    sentiment_score: float = 0.0  # -1 to +1

    @property
    def content_hash(self) -> str:
        return hashlib.md5(
            (self.title + self.content[:200]).encode()
        ).hexdigest()[:12]


# ---- News Sources ----

class CaiLianSheSource:
    """财联社 news source adapter.

    Endpoint (AKShare): ak.stock_telegraph_cls()
    """

    async def fetch(self, count: int = 50) -> list[NewsItem]:
        """Fetch latest news from CaiLianShe."""
        try:
            import akshare as ak
            df = await asyncio.to_thread(ak.stock_telegraph_cls)
            if df is None or df.empty:
                return []

            items: list[NewsItem] = []
            for _, row in df.head(count).iterrows():
                items.append(NewsItem(
                    title=str(row.get("title", row.get("content", ""))),
                    content=str(row.get("content", "")),
                    source="caileianshe",
                    pub_time=str(row.get("ctime", "")),
                ))
            return items
        except Exception as e:
            logger.warning("CaiLianShe fetch failed", error=str(e))
            return []


class EastMoneyNewsSource:
    """东方财富新闻 source adapter."""

    async def fetch(self, count: int = 50) -> list[NewsItem]:
        """Fetch latest stock news."""
        try:
            import akshare as ak
            df = await asyncio.to_thread(ak.stock_news_em)
            if df is None or df.empty:
                return []

            items: list[NewsItem] = []
            for _, row in df.head(count).iterrows():
                items.append(NewsItem(
                    title=str(row.get("title", "")),
                    content=str(row.get("content", "")),
                    source="eastmoney",
                    pub_time=str(row.get("datetime", "")),
                ))
            return items
        except Exception as e:
            logger.warning("EastMoney news fetch failed", error=str(e))
            return []


# ---- News Aggregator ----

class NewsAggregator:
    """Aggregates news from multiple sources with deduplication."""

    def __init__(self) -> None:
        self._sources = [
            CaiLianSheSource(),
            EastMoneyNewsSource(),
        ]

    async def fetch_all(self, count_per_source: int = 30) -> list[NewsItem]:
        """Fetch from all sources in parallel, deduplicate."""
        tasks = [s.fetch(count_per_source) for s in self._sources]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        seen_hashes: set[str] = set()
        unique: list[NewsItem] = []

        for result in results:
            if isinstance(result, Exception):
                logger.warning("News source failed", error=str(result))
                continue
            for item in result:
                h = item.content_hash
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    unique.append(item)

        logger.info("News aggregated", total=len(unique), sources=len(self._sources))
        return unique

    async def fetch_for_symbol(self, symbol: str, count: int = 20) -> list[NewsItem]:
        """Fetch news related to a specific stock."""
        all_items = await self.fetch_all(count)
        # Filter by symbol mention (simple keyword match)
        # In production, use NER or stock code extraction
        related = [
            item for item in all_items
            if symbol in item.title or symbol in item.content
        ]
        return related[:count]


# ---- Sentiment Analysis ----

class SentimentAnalyzer:
    """Chinese financial text sentiment analysis.

    Uses bardsai/finance-sentiment-zh-base (BERT, 0.1B params, 135 samples/sec on GPU).
    Falls back to rule-based keyword matching when model not available.
    """

    POSITIVE_KEYWORDS = [
        "利好", "增长", "突破", "涨停", "增持", "回购", "业绩",
        "超预期", "中标", "签约", "扩产", "分红", "利好",
    ]
    NEGATIVE_KEYWORDS = [
        "利空", "下降", "跌停", "减持", "亏损", "违规", "处罚",
        "退市", "暴雷", "诉讼", "冻结", "质押", "核销",
    ]

    def __init__(self, model_name: str = "bardsai/finance-sentiment-zh-base") -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._use_ml = False

    async def _init_model(self) -> None:
        """Lazy-load the BERT model (heavy, only load once)."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            self._tokenizer = await asyncio.to_thread(
                AutoTokenizer.from_pretrained, self.model_name,
            )
            self._model = await asyncio.to_thread(
                AutoModelForSequenceClassification.from_pretrained, self.model_name,
            )
            self._use_ml = True
            logger.info("Sentiment model loaded", model=self.model_name)
        except Exception as e:
            logger.warning("Failed to load ML model, using rule-based fallback", error=str(e))
            self._use_ml = False

    async def analyze(self, text: str) -> float:
        """Analyze sentiment of a single text. Returns score in [-1, +1]."""
        if self._use_ml and self._model is not None:
            return await self._analyze_ml(text)
        return self._analyze_rule(text)

    async def analyze_batch(self, texts: list[str]) -> list[float]:
        """Batch sentiment analysis."""
        if self._use_ml and self._model is not None:
            return await self._analyze_batch_ml(texts)
        return [self._analyze_rule(t) for t in texts]

    async def _analyze_ml(self, text: str) -> float:
        """ML-based sentiment (BERT)."""
        import torch
        tokens = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        )
        with torch.no_grad():
            outputs = self._model(**tokens)
        logits = outputs.logits[0]
        # Map logits to [-1, +1]
        probs = torch.softmax(logits, dim=-1)
        score = float(probs[1] - probs[0])  # positive - negative
        return score

    async def _analyze_batch_ml(self, texts: list[str]) -> list[float]:
        """ML-based batch sentiment."""
        scores: list[float] = []
        for text in texts:
            scores.append(await self._analyze_ml(text))
        return scores

    @staticmethod
    def _analyze_rule(text: str) -> float:
        """Rule-based keyword matching fallback."""
        text_lower = text.lower()
        pos_count = sum(1 for kw in SentimentAnalyzer.POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in SentimentAnalyzer.NEGATIVE_KEYWORDS if kw in text_lower)
        total = pos_count + neg_count
        if total == 0:
            return 0.0

        # Normalize to [-1, +1] with confidence based on keyword count
        raw = (pos_count - neg_count) / total
        confidence = min(total / 10.0, 1.0)  # More keywords = more confident
        return raw * confidence


# ---- Global Instances ----

_aggregator: Optional[NewsAggregator] = None
_analyzer: Optional[SentimentAnalyzer] = None


def get_aggregator() -> NewsAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = NewsAggregator()
    return _aggregator


def get_sentiment_analyzer() -> SentimentAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer
