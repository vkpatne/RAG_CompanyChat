import os
from core.logger import logger
from core.errors import RetrievalError, GenerationError

from services.llm_manager import LLMManager
from services.faiss_manager import FAISSManager
from services.reranker import rerank_documents
from services.retriever_wrapper import RerankRetriever

from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LangChainService:
	def __init__(self):
		# compose managers
		self.faiss_manager = FAISSManager()
		self.llm_manager = LLMManager()
		# convenience handles
		self.embeddings = self.faiss_manager.embeddings

	def rebuild_index(self):
		"""Force rebuild of FAISS index (manual trigger)."""
		self.faiss_manager.rebuild_index()

	def query(self, question, top_k=1):
		# ensure faiss
		faiss_store = self.faiss_manager.get_faiss_store()
		# ensure llm
		if self.llm_manager.llm is None:
			self.llm_manager.load_llm()
		if self.llm_manager.llm is None:
			if self.llm_manager.llm_load_failed:
				logger.error("LLM not loaded. Check llama-cpp-python installation and model path.")
			raise GenerationError("LLM not loaded. Please check your model path and llama-cpp-python installation.")

		# first-stage retrieval
		candidate_k = max(top_k * 3, top_k)
		try:
			candidates = faiss_store.similarity_search(question, k=candidate_k)
		except Exception as e:
			logger.error(f"FAISS search failed: {e}")
			raise RetrievalError("Retrieval failed: " + str(e))

		# rerank
		ranked_docs = rerank_documents(question, candidates, self.embeddings, top_k)

		# wrap and run RetrievalQA
		retriever = RerankRetriever(ranked_docs)
  
		# ---- Build prompt ----
		prompt = ChatPromptTemplate.from_template(
			"Answer the question using the following context:\n{context}\n\nQuestion: {question}"
		)

		# ---- Create the pipeline ----
		retrieval_chain = (
			{"context": retriever, "question": RunnablePassthrough()}
			| prompt
			| self.llm_manager.llm
			| StrOutputParser()
		)

		# ---- Run it ----
		result = retrieval_chain.invoke(question)
		return result