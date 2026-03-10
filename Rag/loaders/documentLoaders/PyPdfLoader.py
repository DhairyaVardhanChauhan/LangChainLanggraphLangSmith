from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("DhairyaVardhanChauhanResume.pdf")

docs = loader.load()

print(docs[0])