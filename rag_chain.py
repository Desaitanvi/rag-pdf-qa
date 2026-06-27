import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ingest import load_vectorstore

load_dotenv()

def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1
    )

def format_docs(docs):
    """Format retrieved documents into single string"""
    return "\n\n".join([
        f"[Page {doc.metadata.get('page', '?')+1}]\n{doc.page_content}"
        for doc in docs
    ])

def build_rag_chain():
    """Build full RAG chain: retriever | prompt | LLM | parser"""

    # Load vectorstore & retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:""")

    llm = get_llm()
    parser = StrOutputParser()

    # LCEL chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | parser
    )

    return rag_chain, retriever

def get_answer_with_sources(question: str):
    """Get answer + source pages"""
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Get source docs
    source_docs = retriever.invoke(question)

    # Build chain and get answer
    rag_chain, _ = build_rag_chain()
    answer = rag_chain.invoke(question)

    # Extract page numbers
    pages = list(set([
        doc.metadata.get('page', 0) + 1
        for doc in source_docs
    ]))

    return answer, pages, source_docs