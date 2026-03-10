from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai.types import vector_store
from youtube_transcript_api import YouTubeTranscriptApi
load_dotenv()

# LLM
llm_client = ChatAnthropic(model_name="claude-sonnet-4-6")

# Fetch transcript
ytt_api = YouTubeTranscriptApi()
video_id = str(input("Please enter your video id: "))
transcript = ytt_api.fetch(video_id)

# Combine transcript
full_text = " ".join([t.text for t in transcript])

# Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
)

chunks = text_splitter.split_text(full_text)
docs = [
    Document(
        page_content=chunk,
        metadata={"video_id": video_id}
    )
    for chunk in chunks
]
print(docs)
vector_store = Chroma(
    collection_name="test-collection",
    embedding_function=MistralAIEmbeddings(model="mistral-embed" )
)
vector_store.add_documents(docs)

# search and return top k

while(True):
    question = str(input("Please enter you question: "))
    if question == "exit":
        break
    retriever_custom_mmr = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 5, 'fetch_k': 50, 'lambda_mult': 0.25}
    )



    template = """
    You are a helpful assistant answering questions based ONLY on the provided context.
    
    If the answer is not contained in the context, say:
    "I could not find the answer in the provided transcript."
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"],
    )


    retrieved_docs = retriever_custom_mmr.invoke(question)
    for retrieved_doc in retrieved_docs:
        print(retrieved_doc.page_content)
        print("============================")
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    final_prompt = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm_client.invoke(final_prompt)

    print("\nAnswer:\n")
    print(response.content)