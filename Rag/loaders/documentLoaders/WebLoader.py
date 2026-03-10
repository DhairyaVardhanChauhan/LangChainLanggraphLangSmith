from langchain_community.document_loaders import WebBaseLoader

url = "https://docs.langchain.com/oss/python/langchain/install#install-langchain"
loader = WebBaseLoader(url)
docs =  loader.load()

print(docs[0])