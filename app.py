import streamlit as st
import os
import uuid 
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from typing import List, Dict, Any, Optional
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# --- Configuration ---
# Load from environment variables or replace placeholders
GOOGLE_API_KEY =  os.getenv('GOOGLE_API_KEY') 
PINECONE_API_KEY = os.getenvt('PINECONE_API_KEY') 


EMBED_MODEL_NAME = "models/text-embedding-004"
LLM_MODEL_NAME = "gemini-2.0-flash" 
INDEX_NAME = "personal-assistant"

MAX_HISTORY_TURNS = 10
MAX_HISTORY_MESSAGES = MAX_HISTORY_TURNS * 2

langchain_chat_store = {}

def get_full_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in langchain_chat_store:
        langchain_chat_store[session_id] = ChatMessageHistory()
    return langchain_chat_store[session_id]

# --- Langchain Components Setup (Cached for performance) ---

@st.cache_resource
def get_embeddings_model():
    return GoogleGenerativeAIEmbeddings(
        google_api_key=GOOGLE_API_KEY,
        model=EMBED_MODEL_NAME
    )

@st.cache_resource
def get_bm25_encoder():
    return BM25Encoder().default()

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=1.0
    )

class SafeHybridSearchRetriever(PineconeHybridSearchRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        if not self.sparse_encoder:
             return self._dense_only_search(query)
        try:
            sparse_vec = self.sparse_encoder.encode_queries(query)
            if not sparse_vec or not sparse_vec.get("indices") or not sparse_vec.get("values"):
                 return self._dense_only_search(query)
            return super()._get_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            # st.warning(f"Hybrid search failed: {e}. Falling back to dense.") # Can be noisy
            return self._dense_only_search(query)

    def _dense_only_search(self, query: str) -> List[Document]:
        try:
            embedding = self.embeddings.embed_query(query) # self.embeddings should be from parent class
            results = self.index.query(
                vector=embedding, top_k=self.top_k, include_metadata=True, namespace=self.namespace
            )
            return self._process_pinecone_results(results)
        except Exception as e:
            st.error(f"Dense-only search failed: {e}")
            return []

    def _process_pinecone_results(self, results):
        docs = []
        if results and hasattr(results, 'matches'):
             for result in results.matches:
                metadata = result.metadata or {}
                page_content = metadata.pop("text", f"Content missing for ID: {result.id}")
                if page_content.startswith("Content missing"):
                    st.warning(f"Document ID {result.id} has no 'text' in metadata.")
                doc = Document(page_content=page_content, metadata=metadata)
                docs.append(doc)
        return docs


@st.cache_resource
def get_retriever():
    embeddings_model = get_embeddings_model()
    bm25_enc = get_bm25_encoder()
    pinecone_index = get_pinecone_index()
    return SafeHybridSearchRetriever(
        embeddings=embeddings_model,
        sparse_encoder=bm25_enc,
        index=pinecone_index,
        top_k=3
    )

@st.cache_resource
def get_conversational_rag_chain():
    llm = get_llm()
    retriever = get_retriever()

    # 1. Contextualize Question Prompt
    retriever_prompt_template = (
        "Given the recent chat history (up to {max_turns} turns) and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    ).format(max_turns=MAX_HISTORY_TURNS)
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", retriever_prompt_template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    # 2. History-Aware Retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # 3. Question Answering Prompt
    qa_system_prompt = (
    """You are Amaan Poonawala — a witty, self-aware, slightly sarcastic professional AI chatbot designed to answer questions about Amaan's work, projects, and background.
You *must never* reveal that you're an AI language model, or mention Gemini, Google, or any underlying technology. If someone asks what you are, say: "I'm a personal assistant developed by Amaan Poonawala."
Always speak in first person, as if you *are* Amaan himself. You're not talking to Amaan — you're talking to someone *asking about Amaan*. So answer like it's *you* they're asking about.
Keep it natural, confident, with a sprinkle of humor."""
"Remeber: Don't tolerate or entertain any query not related to Amaan's work, projects, or background. "
    "Answer the user's questions based on the retrieved context below and the recent chat history (up to {max_turns} turns).\n"
    "If the context doesn't contain the answer, make something relevant with the given documents.\n\n"
    "Context:\n{{context}}\n\n"
    "Recent Chat History:\n{{chat_history}}"
).format(max_turns=MAX_HISTORY_TURNS)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ])

    # 4. Question Answering Chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # 5. Combined RAG Chain
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # 6. Wrapper Function to Limit History
    def limit_history_for_rag_chain(input_dict: Dict[str, Any]) -> Dict[str, Any]:
        modified_input = input_dict.copy()
        if "chat_history" in modified_input:
            history = modified_input["chat_history"]
            if isinstance(history, list) and all(isinstance(m, BaseMessage) for m in history):
                limited_history = history[-MAX_HISTORY_MESSAGES:]
                modified_input["chat_history"] = limited_history
        return modified_input

    # 7. Final Conversational Chain
    conversational_chain = RunnableWithMessageHistory(
        runnable=RunnableLambda(limit_history_for_rag_chain) | rag_chain,
        get_session_history=get_full_session_history, # Uses the global store
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_chain

# --- Streamlit App UI ---
st.set_page_config(page_title="Personal Assistant Chatbot", layout="wide")
st.title("🤖 Personal Assistant Chatbot")
st.caption("Powered by Langchain, Pinecone, and Google Gemini")

# Initialize or get Langchain session ID for the current Streamlit user session
if "langchain_session_id" not in st.session_state:
    st.session_state.langchain_session_id = str(uuid.uuid4())
    st.info(f"New Langchain session started: {st.session_state.langchain_session_id}. Chat history will be maintained for this session.")


# Initialize chat history for Streamlit display
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "hi, I am a personal assitant developed by Amaan Poonawala. I am here to assist you with your queries regarding Amaan's work, projects, and background. so don't even try to ask me some stupid random stuff like what is 2 + 2, or what is the weight of the sun. so keeping that in mind, how can I help you today?"}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "documents" in msg and msg["documents"]:
            with st.expander("Retrieved Documents"):
                for i, doc in enumerate(msg["documents"]):
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(f"Page Content (snippet): {doc.page_content[:250]}...")
                    st.caption(f"Metadata: {doc.metadata}")


# Get user input
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the conversational RAG chain
    try:
        chain = get_conversational_rag_chain()
    except Exception as e:
        st.error(f"Failed to initialize the RAG chain: {e}")
        print(f"Error: {e}")
        st.stop()


    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response_content = ""
        retrieved_docs_for_display = []

        with st.spinner("Thinking..."):
            try:
                response = chain.invoke(
                    {"input": prompt},
                    config={"configurable": {"session_id": st.session_state.langchain_session_id}}
                )
                full_response_content = response.get("answer", "Sorry, I encountered an issue and couldn't get an answer.")
                retrieved_docs_for_display = response.get("context", [])

            except Exception as e:
                st.error(f"Error during chain invocation: {e}")
                full_response_content = "An error occurred while processing your request."

        message_placeholder.markdown(full_response_content)
        if retrieved_docs_for_display:
            with st.expander("Retrieved Documents For This Response"):
                for i, doc in enumerate(retrieved_docs_for_display):
                    st.markdown(f"**Document {i+1}:**")
                    st.markdown(f"Page Content (snippet): {doc.page_content[:250]}...")
                    st.caption(f"Metadata: {doc.metadata}")


    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response_content,
        "documents": retrieved_docs_for_display # Store documents with the message for redisplay
    })