from langchain_core.retrievers import BaseRetriever

class RerankRetriever(BaseRetriever):
	# Provide common BaseRetriever attributes so LangChain accessors don't fail
	tags: list = None
	metadata: dict = None

	def __init__(self, docs, **kwargs):
		# call BaseRetriever initializer (pydantic) with any extra kwargs
		try:
			super().__init__(**kwargs)
		except Exception:
			try:
				super().__init__()
			except Exception:
				pass
		self._docs = docs

	def get_relevant_documents(self, query_input):
		return self._docs

	async def aget_relevant_documents(self, query_input):
		return self._docs
