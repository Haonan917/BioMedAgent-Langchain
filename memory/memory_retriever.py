import json
from typing import Any, Optional

import redis
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_redis import RedisVectorStore

from config import Config


class RedisMemoryRetriever:
    def __init__(
        self,
        redis_url: Optional[str] = None,
        index_name: str = "agent_memory",
        embedding_model: str = "text-embedding-3-small",
        threshold: float = 0.4,
    ) -> None:
        self.config = Config()

        self.redis_url = (
            redis_url
            or getattr(self.config, "REDIS_URL", None)
            or "redis://localhost:6379/0"
        )
        self.index_name = index_name
        self.threshold = threshold
        # 继续使用 Redis client 做日志
        self.r = redis.Redis.from_url(self.redis_url, decode_responses=True)
        # Embedding model
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

        # LangChain Redis vector store
        self.vector_store = RedisVectorStore(
            index_name=self.index_name,
            embeddings=self.embeddings,
            redis_url=self.redis_url,
        )

    def add_memory(
        self,
        item_key: str,
        value: str,
        key: Optional[str] = None,
        memory_id: Optional[str] = None,
        question_id: Optional[str] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        写入一条 memory。

        对应你旧结果里的字段：
            item["value"]
            item["key"]
            item["id"]
            item["question_id"]

        新版把它们放到 Document.metadata 里。
        """

        metadata = {
            "item_key": item_key,
            "key": key or "",
            "memory_id": memory_id or "",
            "question_id": question_id or "",
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        doc = Document(
            page_content=value,
            metadata=metadata,
        )

        ids = self.vector_store.add_documents([doc])

        return ids[0]

    def add_memories(self, memories: list[dict[str, Any]]) -> list[str]:
        """
        memories
        [
            {
                "item_key": "blast",
                "value": "BLAST is useful for sequence similarity search.",
                "key": "old question text",
                "memory_id": "xxx",
                "question_id": "q1"
            }
        ]
        """

        docs = []
        for memory in memories:
            docs.append(
                Document(
                    page_content=memory["value"],
                    metadata={
                        "item_key": memory.get("item_key", ""),
                        "key": memory.get("key", ""),
                        "memory_id": memory.get("memory_id", memory.get("id", "")),
                        "question_id": memory.get("question_id", ""),
                    },
                )
            )

        return self.vector_store.add_documents(docs)

    def _similarity_search_with_score(
        self,
        item_key: str,
        query_key: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        """
        """

        try:
            results = self.vector_store.similarity_search_with_score(
                query=query_key,
                k=k,
                filter={"item_key": item_key},
            )
        except TypeError:
            # 兼容某些版本 / 配置下 filter 参数不可用的情况
            raw_results = self.vector_store.similarity_search_with_score(
                query=query_key,
                k=k * 5,
            )
            results = [
                (doc, score)
                for doc, score in raw_results
                if doc.metadata.get("item_key") == item_key
            ][:k]

        return results

    def _score_passes_threshold(self, score: float) -> bool:
        """
        """
        return True

    def _log_memory_hit(
        self,
        item_key: str,
        current_query_key: str,
        doc: Document,
        score: float,
        total_task_id: str,
        result_count: int,
    ) -> None:
        """
        保留你旧代码的日志结构。
        """

        metadata = doc.metadata or {}

        self.r.lpush(
            self.config.REDIS_MEMORY_LOG2,
            json.dumps(
                {
                    "item_key": item_key,
                    "current_query_key": current_query_key,
                    "origin_query_key": metadata.get("key", ""),
                    "query_value": doc.page_content,
                    "memory_id": metadata.get("memory_id", ""),
                    "origin_question_id": metadata.get("question_id", ""),
                    "current_task_id": total_task_id,
                    "k": result_count,
                    "score": score,
                },
                ensure_ascii=False,
            ),
        )

    def match(
        self,
        item_key: str,
        query_key: str,
        total_task_id: str,
        k: int = 3,
    ) -> str:
        """
        旧接口兼容版。

        item_key:
            tool name / agent name / memory category

        query_key:
            当前问题文本，旧代码里叫 query_key

        total_task_id:
            当前任务 ID，用于日志

        k:
            返回 memory 数量
        """

        results = self._similarity_search_with_score(
            item_key=item_key,
            query_key=query_key,
            k=k,
        )

        results = [
            (doc, score)
            for doc, score in results
            if self._score_passes_threshold(score)
        ]

        if len(results) == 0:
            self.config.log_error("special", "result is None")
            return ""

        content = []

        for doc, score in results:
            content.append(doc.page_content)

            self._log_memory_hit(
                item_key=item_key,
                current_query_key=query_key,
                doc=doc,
                score=score,
                total_task_id=total_task_id,
                result_count=len(results),
            )

        content_text = f"\n{'=' * 10}\n".join(content)

        response = f"""
            Here's a summary of some of your experiences and memories that you can use for reference: 【
            {content_text}
            】You must learn from experience and memory as well as reference!
        """

        return response
    def retrieve_documents(
    self,
    item_key: str,
    query_key: str,
    k: int = 3,
) -> list[Document]:
        results = self._similarity_search_with_score(
            item_key=item_key,
            query_key=query_key,
            k=k,
        )

        return [doc for doc, score in results]


    def format_memory_context(self, docs: list[Document]) -> str:
        if not docs:
            return ""

        content_text = f"\n{'=' * 10}\n".join(
            doc.page_content for doc in docs
        )

        return f"""
            Here's a summary of some of your experiences and memories that you can use for reference: 【
            {content_text}
            】You must learn from experience and memory as well as reference!
        """