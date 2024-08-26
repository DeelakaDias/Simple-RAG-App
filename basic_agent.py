from llama_index.llms.groq import Groq 
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.yahoo_finance import YahooFinanceToolSpec


MODEL_NAME = "llama-3.1-70b-versatile"
API_KEY = "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"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
KNOWLEDGE_SOURCE_PATH = "./data2/"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
OUTPUT_TOKENS = 512

def get_llm(model_name, api_key):
    return Groq(model= model_name, api_key=api_key)

def initialize_settings():
    Settings.llm = get_llm(MODEL_NAME, API_KEY)
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
    Settings.num_output = OUTPUT_TOKENS
    Settings.node_parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def multiply(a: float, b: float):
    return a * b 

def add(a: float, b: float):
    return a + b 

multiply_tool = FunctionTool.from_defaults(fn=multiply)
add_tool = FunctionTool.from_defaults(fn=add)

def load_index(folder_path):
    documents = SimpleDirectoryReader(folder_path).load_data()
    initialize_settings()
    index = VectorStoreIndex(documents, embed_model=Settings.embed_model, llm = Settings.llm)
    index.storage_context.persist()
    return index.as_query_engine()

query_engine = load_index(KNOWLEDGE_SOURCE_PATH)

budget_tool = QueryEngineTool.from_defaults(query_engine, name="canadian_budget_2023",
                                            description="A RAG engine with some basic facts about the 2023 Canadian federal budget.")

#initialize_settings()
finance_tools = YahooFinanceToolSpec().to_tool_list()
finance_tools.extend([multiply_tool, add_tool])

agent = ReActAgent.from_tools([budget_tool, multiply_tool, add_tool], llm=Settings.llm, verbose=True)

response = agent.chat("What is the total amout of the 2023 Canadian federal budget multiplied by 3? Go step by step, using a tool to do any math.")
#response = agent.chat("What is the current price of NVDA?")

print(response)