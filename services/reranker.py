import math
from core.logger import logger

def _cosine_sim(a, b):
	# simple cosine (handles zero vectors)
	adotb = sum(x*y for x,y in zip(a,b))
	norm_a = math.sqrt(sum(x*x for x in a))
	norm_b = math.sqrt(sum(x*x for x in b))
	if norm_a == 0 or norm_b == 0:
		return 0.0
	return adotb / (norm_a * norm_b)

def rerank_documents(query: str, candidates: list, embeddings, top_k: int):
	"""
	Compute embeddings for query and candidate docs and return candidates
	sorted by cosine similarity (descending). `embeddings` should implement
	embed_query and embed_documents.
	"""
	if not candidates:
		return []

	try:
		query_vec = embeddings.embed_query(query)
		texts = [getattr(d, "page_content", "") or str(d) for d in candidates]
		doc_vecs = embeddings.embed_documents(texts)

		scored = []
		for doc, vec in zip(candidates, doc_vecs):
			score = _cosine_sim(query_vec, vec)
			scored.append((score, doc))

		scored.sort(key=lambda x: x[0], reverse=True)
		top_docs = [doc for _, doc in scored[:top_k]]
		logger.info(f"Reranked {len(candidates)} candidates -> selected {len(top_docs)}")
		return top_docs
	except Exception as e:
		logger.warning(f"Rerank failed, falling back to original order: {e}")
		return candidates[:top_k]
