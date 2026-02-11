import streamlit as st
import requests
import json

# Streamlit page configuration
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="🤖",
    layout="centered"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def call_ollama_chat_stream(messages, model="deepseek-r1:1.5b"):
    """Call Ollama chat API with streaming responses"""
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": True
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                json_response = json.loads(line)
                if "message" in json_response:
                    chunk = json_response["message"]["content"]
                    yield chunk
                    
                # Check if done
                if json_response.get("done", False):
                    break
                    
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to Ollama: {str(e)}\n\nMake sure Ollama is running with: ollama serve"

def call_ollama_chat(messages, model="deepseek-r1:1.5b"):
    """Call Ollama chat API without streaming"""
    url = "http://localhost:11434/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"Error connecting to Ollama: {str(e)}\n\nMake sure Ollama is running with: ollama serve"

# App title and description
st.title("🤖 DeepSeek Chatbot")
st.caption("Powered by Ollama DeepSeek-R1 1.5B")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_name = st.text_input("Model Name", value="deepseek-r1:1.5b")
    
    # Streaming toggle
    use_streaming = st.checkbox("Enable Streaming", value=True, 
                                 help="Show responses word-by-word as they're generated")
    
    # Temperature slider (if you want to add it)
    # temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    # Info section
    st.markdown("### ℹ️ Information")
    st.caption("Make sure Ollama is running:")
    st.code("ollama serve", language="bash")
    st.caption("Verify model is available:")
    st.code("ollama list", language="bash")
    
    # Message count
    st.divider()
    st.metric("Messages", len(st.session_state.messages))

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get bot response
    with st.chat_message("assistant"):
        if use_streaming:
            # Streaming response
            message_placeholder = st.empty()
            full_response = ""
            
            for chunk in call_ollama_chat_stream(st.session_state.messages, model=model_name):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        else:
            # Non-streaming response
            with st.spinner("Thinking..."):
                full_response = call_ollama_chat(st.session_state.messages, model=model_name)
                st.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})