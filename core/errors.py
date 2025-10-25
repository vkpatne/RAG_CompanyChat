class RAGError(Exception): pass
class IngestionError(RAGError): pass
class RetrievalError(RAGError): pass
class GenerationError(RAGError): pass
