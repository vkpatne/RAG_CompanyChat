
import os, pickle
from core.logger import logger

class IndexManager:
    def __init__(self, docs_path='docs.pkl'):
        self.docs_path = docs_path
        self.docs = []
        self.load()

    def load(self):
        if os.path.exists(self.docs_path):
            with open(self.docs_path, 'rb') as f:
                self.docs = pickle.load(f)
            logger.info(f'Loaded {len(self.docs)} docs from {self.docs_path}')

    def save(self):
        with open(self.docs_path, 'wb') as f:
            pickle.dump(self.docs, f)
        logger.info(f'Saved {len(self.docs)} docs to {self.docs_path}')

    def add_chunks(self, chunks):
        start = len(self.docs)
        for i, c in enumerate(chunks):
            self.docs.append({'id': start + i, 'text': c})
        self.save()

    def get_texts(self):
        return [d['text'] for d in self.docs]
