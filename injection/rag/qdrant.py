import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain.docstore.document import Document
from langchain.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.conversions import common_types
from qdrant_client.models import Distance, VectorParams

from embedding import LLamaEmbedding

from loguru import logger

DictFilter = Dict[str, Union[str, int, bool, dict, list]]
MetadataFilter = Union[DictFilter, common_types.Filter]


class CustomQdrant(Qdrant):
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        search_params: Optional[common_types.SearchParams] = None,
        offset: int = 0,
        score_threshold: Optional[float] = None,
        consistency: Optional[common_types.ReadConsistency] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            search_params: Additional search params
            offset:
                Offset of the first result to return.
                May be used to paginate results.
                Note: large offset values may cause performance issues.
            score_threshold:
                Define a minimal score threshold for the result.
                If defined, less similar results will not be returned.
                Score of the returned result might be higher or smaller than the
                threshold depending on the Distance function used.
                E.g. for cosine similarity only higher scores will be returned.
            consistency:
                Read consistency of the search. Defines how many replicas should be
                queried before returning the result.
                Values:
                - int - number of replicas to query, values should present in all
                        queried replicas
                - 'majority' - query all replicas, but return values present in the
                               majority of replicas
                - 'quorum' - query the majority of replicas, return values present in
                             all of them
                - 'all' - query all replicas, and return values present in all replicas

        Returns:
            List of documents most similar to the query text and cosine
            distance in float for each.
            Lower score represents more similarity.
        """

        if filter is not None and isinstance(filter, dict):
            warnings.warn(
                "Using dict as a `filter` is deprecated. Please use qdrant-client "
                "filters directly: "
                "https://qdrant.tech/documentation/concepts/filtering/",
                DeprecationWarning,
            )
            qdrant_filter = self._qdrant_filter_from_dict(filter)
        else:
            qdrant_filter = filter
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=self._embed_query(query),
            query_filter=qdrant_filter,
            search_params=search_params,
            limit=k,
            offset=offset,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
            consistency=consistency,
            **kwargs,
        )
        return [
            (
                self._document_from_scored_point(
                    result, self.content_payload_key, self.metadata_payload_key
                ),
                result.score,
                result.id,
            )
            for result in results
        ]

    def delete_vectors(self, vectors_ids: List[str]):
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=vectors_ids,
        )


def load_qdrant(collection_name, vector_size, embeddings):
    url = os.env["QDRANT_DB_URI"]
    api_key = os.env["QDRANT_API_KEY"]

    qdrant_client = QdrantClient(
        url=url,
        api_key=api_key,
        prefer_grpc=True,
    )

    logger.info(
        f"Collection {collection_name} with documents {qdrant_client.get_collection(collection_name=collection_name)}"
    )

    qdrant = CustomQdrant(
        client=qdrant_client, collection_name=collection_name, embeddings=embeddings
    )

    try:
        qdrant_client.get_collection(collection_name=collection_name)
    except:
        qdrant_client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

    return qdrant


class QdrantHandler:
    def __init__(
        self, qdrant_collection, vector_size, embeddings
    ):
        self.qdrant = load_qdrant(qdrant_collection, vector_size, embeddings)

        self.qdrant_collection = qdrant_collection
        self.vector_size = vector_size
        self.embeddings = embeddings

    def values(self):
        self.connection_check()
        return self.qdrant, self.qdrant_images

    def connection_check(self):
        try:
            self.qdrant.client.get_collection(collection_name=self.qdrant_collection)
            logger.debug(f"Text qdrant is working")
        except Exception as e:
            logger.warning(f"Text qdrant connection was DOWN, trying to reconnect: {e}")
            self.qdrant = load_qdrant(
                self.qdrant_collection, self.vector_size, self.embeddings
            )
            logger.debug("Reconnect to text qdrant successfully")


class AllQdrantHandler:
    def __init__(self):
        self.qdrant_opensource = QdrantHandler(
            Config.OSS_QDRANT_COLLECTION,
            embeddings=LLamaEmbedding(),
        )

    def open_ai(self):
        return self.qdrant_openai.values()

    def open_source(self):
        return self.qdrant_opensource.values()

    def __getitem__(self, model_type: str):
        if model_type == "OPEN_AI":
            return self.open_ai()
        elif model_type == "OPEN_SOURCE":
            return self.open_source()
        else:
            raise ValueError(f"Model type {model_type} not supported")
