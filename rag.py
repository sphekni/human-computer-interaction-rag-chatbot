import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class RAGChatbot:
    def __init__(self, knowledge_file="data/sample_knowledge.txt"):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        with open(knowledge_file, "r", encoding="utf-8") as f:
            self.documents = [line.strip() for line in f if line.strip()]

        self.doc_embeddings = self.embedder.encode(self.documents)
        self.index = faiss.IndexFlatL2(self.doc_embeddings.shape[1])
        self.index.add(np.array(self.doc_embeddings).astype("float32"))

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    def retrieve(self, query, k=3):
        query_vec = self.embedder.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype("float32"), k)
        return [self.documents[i] for i in indices[0]]

    def generate(self, query):
        context_docs = self.retrieve(query)
        context = "\n".join(context_docs)

        prompt = (
            "You are an assistant that answers based on the context.\n\n"
            f"Context:\n{context}\n\n"
            f"User: {query}\nAssistant:"
        )

        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(
            inputs,
            max_length=200,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            top_k=50,
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)