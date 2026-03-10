from langchain_community.document_loaders import TextLoader

loader = TextLoader("poem.txt",encoding="utf-8")

res = loader.load()
print(res[0])