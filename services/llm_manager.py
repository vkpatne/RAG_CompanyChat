import os
from langchain_community.llms import LlamaCpp
from core.logger import logger
from core.config import LLM_LOCAL_PATH

class LLMManager:
	"""Manage loading and reloading of a local LlamaCpp model."""
	def __init__(self):
		self.llm = None
		self.llm_load_failed = False
		self.load_llm()

	def load_llm(self):
		"""Load LlamaCpp model into memory. No-op if already loaded."""
		if self.llm is not None:
			return
		try:
			self.llm = LlamaCpp(
				model_path=LLM_LOCAL_PATH,
				n_threads=os.cpu_count(),
				n_ctx=2048,
				n_batch=256,
				temperature=0.3,
				max_tokens=256,
				verbose=False,
				use_mlock=True,
				use_mmap=True,
				n_gpu_layers=0,
			)
			self.llm_load_failed = False
			logger.info(f"Loaded LlamaCpp LLM from {LLM_LOCAL_PATH}")
		except Exception as e:
			logger.error(
				f"Failed to load GGUF model: {e}\n"
				f"Check that llama-cpp-python is installed, and that your model path is correct and points to a valid .gguf file. "
				f"Current model path: {LLM_LOCAL_PATH}"
			)
			self.llm = None
			self.llm_load_failed = True

	def reload_llm(self):
		"""Force reload the LLM (useful for swapping model files or recovering)."""
		try:
			self.llm = None
			self.llm_load_failed = False
			self.load_llm()
			return {"status": "reloaded" if self.llm is not None else "failed"}
		except Exception as e:
			logger.error(f"LLM reload failed: {e}")
			return {"status": "error", "detail": str(e)}
