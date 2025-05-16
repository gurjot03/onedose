import streamlit as st
from google import genai
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid

load_dotenv()

st.set_page_config(page_icon="💬")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

name = "chat_history"

def setup_qdrant_collection():
    collections = qdrant_client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    
    if name not in collection_names:
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=768, 
                distance=models.Distance.COSINE
            )
        )

def get_embeddings(text):
    result = client.models.embed_content(model="text-embedding-004", contents=text)
    return result.embeddings[0].values

def store_message(role, content, session_id):
    embedding = get_embeddings(content)
    
    qdrant_client.upsert(
        collection_name=name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "role": role,
                    "content": content,
                    "session_id": session_id
                }
            )
        ]
    )

def get_relevant_history(query_text, session_id, top_k):
    query_vector = get_embeddings(query_text)

    search_result = qdrant_client.search(
        collection_name=name,
        query_vector=query_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="session_id",
                    match=models.MatchValue(value=session_id)
                )
            ]
        ),
        limit=top_k,
        with_payload=True
    )
    print(search_result)
    relevant_messages = [
        f"{hit.payload['role']}: {hit.payload['content']}" 
        for hit in search_result
    ]
    return relevant_messages

setup_qdrant_collection()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

def clear_chat():
    qdrant_client.delete(
        collection_name=name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="session_id",
                        match=models.MatchValue(value=st.session_state.session_id)
                    )
                ]
            )
        )
    )
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())

def get_gemini_response(user_input, session_id):
    
    try:
        history = get_relevant_history(user_input, session_id, 5)
        print(history)
        
        prompt = f"""
        You are a helpful assistant with memory of the recent conversation.
        Conversation history:
        {history}
        
        User's latest message: {user_input}
        
        Respond to the user's latest message while considering this conversation context.
        """
        
        response = client.models.generate_content(model="gemini-1.5-flash",contents=prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating response: {str(e)}"

st.title("Chatbot")

if st.button("Clear Chat"):
    clear_chat()
    st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Type your message here..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    store_message("user", prompt, st.session_state.session_id)
    
    with st.chat_message("assistant"):
        response = get_gemini_response(prompt, st.session_state.session_id)
        st.write(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    store_message("assistant", response, st.session_state.session_id)