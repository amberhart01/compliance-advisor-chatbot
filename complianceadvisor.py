import os
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ─────────────────────────────────────────────────────────────────────────────
# 1) LLM & Environment Memory
# ─────────────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4", temperature=0)
embeds = OpenAIEmbeddings()
chroma_env = Chroma(persist_directory="./chromadb", embedding_function=embeds)
memory = VectorStoreRetrieverMemory(
    retriever=chroma_env.as_retriever(),
    memory_key="env_facts"
)

# ─────────────────────────────────────────────────────────────────────────────
# 2) Build & Index NIST SP 800-53
# ─────────────────────────────────────────────────────────────────────────────
nist_url = "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf"
nist_docs = PyPDFLoader(nist_url).load_and_split()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
nist_chunks = splitter.split_documents(nist_docs)
nist_store = Chroma.from_documents(nist_chunks, embeds, persist_directory="./nist_index")
nist_retriever = nist_store.as_retriever(search_kwargs={"k": 3})

# Summarization chain (map-reduce)
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")

# ─────────────────────────────────────────────────────────────────────────────
# 3) NIST-Only Tools & Agents
# ─────────────────────────────────────────────────────────────────────────────
def fetch_nist_summary(question: str) -> str:
    docs = nist_retriever.get_relevant_documents(question)
    return summarize_chain.run(docs)

fetch_tool = Tool(
    name="fetch_nist",
    func=fetch_nist_summary,
    description="Retrieve & summarize relevant NIST SP 800-53 text for the question"
)

def analyze_nist_summary(summary: str) -> str:
    prompt = f"""
You are a Compliance Analyst. Given this NIST policy summary:
{summary}

Provide a concise, actionable answer.
"""
    response: AIMessage = llm([HumanMessage(content=prompt)])
    return response.content

analyze_tool = Tool(
    name="analyze_nist",
    func=analyze_nist_summary,
    description="Analyze the summarized NIST policy for a user question"
)

fetcher_agent   = initialize_agent([fetch_tool],   llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
analyzer_agent  = initialize_agent([analyze_tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)

def nist_only_chat(user_query: str) -> str:
    # 1) Summarize relevant NIST controls
    policy_summary = fetcher_agent.run(user_query)
    analysis       = analyzer_agent.run(policy_summary)

    # 2) Directly ask the LLM for final answer
    prompt = f"""
Policy Analysis:
{analysis}

Question:
{user_query}

Provide a concise, actionable answer.
"""
    response: AIMessage = llm([HumanMessage(content=prompt)])
    return response.content

# ─────────────────────────────────────────────────────────────────────────────
# 4) Build & Index Your Company Policy
# ─────────────────────────────────────────────────────────────────────────────
# (Re-use the same splitter)
def build_company_index(path_or_url: str, index_dir: str):
    if path_or_url.lower().endswith(".pdf"):
        docs = PyPDFLoader(path_or_url).load_and_split()
    else:
        docs = UnstructuredURLLoader(urls=[path_or_url]).load()
    chunks = splitter.split_documents(docs)
    return Chroma.from_documents(chunks, embeds, persist_directory=index_dir)

# example: call once when the policy is updated
policy_store     = build_company_index("SANS_Acceptable_Encryption_Standard_April2025.pdf", "./policy_index")
policy_retriever = policy_store.as_retriever(search_kwargs={"k": 3})

# ─────────────────────────────────────────────────────────────────────────────
# 5) Gap-Analysis Tool & Agent
# ─────────────────────────────────────────────────────────────────────────────
def gap_analysis(question: str) -> str:
    policy_chunks = policy_retriever.get_relevant_documents(question)
    nist_chunks   = nist_retriever.get_relevant_documents(question)

    company_summary = summarize_chain.run(policy_chunks)
    nist_summary    = summarize_chain.run(nist_chunks)

    prompt = f"""
You are a Security Auditor.
Company policy excerpt:
{company_summary}

NIST controls excerpt:
{nist_summary}

Question: {question}

Identify any gaps in the company policy relative to the NIST controls above, and recommend improvements.
"""
    response: AIMessage = llm([HumanMessage(content=prompt)])
    return response.content

gap_tool = Tool(
    name="gap_analysis",
    func=gap_analysis,
    description="Compare company policy vs. NIST controls and report any gaps"
)

gap_agent = initialize_agent(
    [gap_tool], llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

def company_compliance_chat(user_query: str) -> str:
    # 1) Run the gap‐analysis agent
    answer = gap_agent.run(user_query)

    # 2) Return only the gap recommendations
    return answer
# ─────────────────────────────────────────────────────────────────────────────
# 6) CLI Entrypoint (optional)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    q = input("Enter your question: ")
    print("\n--- NIST-Only Response ---")
    print(nist_only_chat(q))
    print("\n--- Company Gap Analysis ---")
    print(company_compliance_chat(q))
