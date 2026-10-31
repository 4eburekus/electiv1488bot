import sys
from importlib.metadata import version, PackageNotFoundError

print(f"Python: {sys.version.split()[0]}  (проверено на 3.12)\n")

packages = [
    "langchain", "langchain-classic", "langchain-community", "langchain-huggingface",
    "langchain-openai", "langchain-chroma", "langchain-tavily", "langgraph",
    "sentence-transformers", "transformers", "torch", "faiss-cpu", "chromadb",
    "ddgs", "pypdf", "pdfplumber", "numpy",
]
missing = []
for name in packages:
    try:
        print(f"✅ {name}: {version(name)}")
    except PackageNotFoundError:
        print(f"❌ {name} не установлен")
        missing.append(name)

import torch
print(f"\n   GPU доступен: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer

if missing:
    print("\n❌ Не хватает пакетов:", ", ".join(missing), "— выполните ячейки установки выше или pip install -r requirements.txt")
else:
    print("\n🎉 Если все ✅ — можно продолжить работу!")