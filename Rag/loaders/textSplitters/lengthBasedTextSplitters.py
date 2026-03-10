from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

text = """Golden hues kiss the sky,Whispering secrets to the dawn.Birds awaken with a sigh,As morning breaks, night withdrawn.Silent trees sway in the breeze,A symphony of nature's ease.Sunlight dances, colors bloom,In the quiet morning's room.The world awakes to light anew,Bathed in morning's gentle dew."""

splitter = CharacterTextSplitter(chunk_size=50,chunk_overlap=0,separator='')
result = splitter.split_text(text)
print(result)


# // loading pdf and splitting

loader = PyPDFLoader("../documentLoaders/DhairyaVardhanChauhanResume.pdf")

docs = loader.load()

res = []
for doc in docs:
    res.extend(splitter.split_text(doc.page_content))

print(res)


# // easier way
docks = splitter.split_documents(docs)
print(res)