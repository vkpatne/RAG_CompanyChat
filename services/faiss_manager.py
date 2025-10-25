import os, json, hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from core.config import EMBED_MODEL_PATH
from storage.index_manager import IndexManager
from core.logger import logger
from core.errors import RetrievalError

class FAISSManager:
	"""Manage FAISS vector store: load from disk, rebuild, and persist metadata hash."""
	def __init__(self, faiss_dir="./faiss_store"):
		self.index_manager = IndexManager()
		self.embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL_PATH)
		self.faiss_dir = faiss_dir
		self.meta_file = os.path.join(self.faiss_dir, "meta.json")
		self.faiss_store = None
		self._load_or_rebuild_faiss()

	def _compute_docs_hash(self):
		"""Compute a lightweight checksum of all docs to detect changes."""
		data = "".join([d["text"] for d in self.index_manager.docs])
		return hashlib.md5(data.encode("utf-8")).hexdigest() if data else None

	def _load_or_rebuild_faiss(self):
		"""Loads existing FAISS if valid, otherwise rebuilds. Skips if no docs."""
		current_hash = self._compute_docs_hash()
		if not self.index_manager.docs:
			logger.info("No documents found at startup. FAISS index will be built after ingestion.")
			self.faiss_store = None
			return

		meta_hash = None
		# Try to load metadata if exists
		if os.path.exists(self.meta_file):
			try:
				with open(self.meta_file, "r") as f:
					meta = json.load(f)
				meta_hash = meta.get("docs_hash")
			except Exception:
				logger.warning("Meta file unreadable, will rebuild FAISS.")

		if (
			os.path.exists(self.faiss_dir)
			and meta_hash
			and current_hash == meta_hash
		):
			try:
				self.faiss_store = FAISS.load_local(self.faiss_dir, self.embeddings)
				logger.info("✅ Loaded existing FAISS vector store from disk.")
				return
			except Exception as e:
				logger.warning(f"Failed to load FAISS: {e}. Rebuilding...")

		# Otherwise rebuild
		self._build_and_save_faiss(current_hash)

	def _build_and_save_faiss(self, docs_hash):
		texts = self.index_manager.get_texts()
		if not texts:
			raise RetrievalError("No documents found. Please ingest first.")
		self.faiss_store = FAISS.from_texts(texts, self.embeddings)
		os.makedirs(self.faiss_dir, exist_ok=True)
		self.faiss_store.save_local(self.faiss_dir)
		# Save hash metadata
		with open(self.meta_file, "w") as f:
			json.dump({"docs_hash": docs_hash, "count": len(texts)}, f)
		logger.info(f"Rebuilt FAISS store with {len(texts)} docs and saved to disk.")

	def get_faiss_store(self):
		if self.faiss_store is None:
			self._load_or_rebuild_faiss()
		return self.faiss_store

	def rebuild_index(self):
		"""Force rebuild of FAISS index (manual trigger)."""
		logger.info(f"Rebuilt index.")
		docs_hash = self._compute_docs_hash()
		self._build_and_save_faiss(docs_hash)