import streamlit as st
import requests
import os
from pathlib import Path

# Add parent directory to path for importing configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings

# Configuration
settings = get_settings()

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = "1"

def load_logo():
    """Load and display company logo."""
    try:
        if os.path.exists(settings.LOGO_PATH):
            st.image(settings.LOGO_PATH, width=150)
        else:
            st.info("🏢 Draiver - Logo not found. Please place the logo at: " + settings.LOGO_PATH)
    except Exception as e:
        st.warning(f"Error loading logo: {str(e)}")

def send_message_to_backend(user_input: str, session_id: str):
    """Send message to backend and return response."""
    try:
        response = requests.post(
            f"{settings.BACKEND_URL}/chat",
            json={
                "session_id": session_id,
                "user_input": user_input
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to backend server. Make sure it's running at {settings.BACKEND_URL}")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ Server took too long to respond. Please try again.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ Server error: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        return None

def clear_conversation():
    """Clear current conversation."""
    try:
        response = requests.delete(
            f"{settings.BACKEND_URL}/chat/{st.session_state.session_id}",
            timeout=10
        )
        response.raise_for_status()
        st.session_state.messages = []
        st.success("🧹 Conversation cleared successfully")
        st.rerun()
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to server to clear conversation")
    except Exception as e:
        st.error(f"❌ Error clearing conversation: {str(e)}")

def main():
    """Main Streamlit application function."""
    # Page configuration
    st.set_page_config(
        page_title="Draiver Virtual Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    initialize_session_state()

    # Custom CSS for dark theme and improved appearance
    st.markdown("""
    <style>
    /* Dark theme background */
    .stApp {
        background-color: #1B1C1E;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1B1C1E;
    }
    
    /* Main content area */
    .main .block-container {
        background-color: #1B1C1E;
        color: white;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        padding: 1rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* User messages - aligned right with light gray background */
    .stChatMessage[data-testid="user-message"] {
        background-color: #4a4a4a;
        margin-left: 20%;
        border-radius: 15px;
        padding: 10px;
    }
    
    /* Assistant messages - aligned left with blue/dark background */
    .stChatMessage[data-testid="assistant-message"] {
        background-color: #2d3748;
        margin-right: 20%;
        border-radius: 15px;
        padding: 10px;
    }
    
    /* Input styling */
    .stChatInput > div > div > div > input {
        border-radius: 20px;
        background-color: #2d3748;
        color: white;
        border: 1px solid #4a5568;
    }
    
    .stChatInput > div > div > div > input::placeholder {
        color: #a0aec0;
    }
    
    /* Sidebar content styling */
    .sidebar .sidebar-content {
        background-color: #1B1C1E;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #4299e1;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.5rem 1rem;
    }
    
    .stButton > button:hover {
        background-color: #3182ce;
    }
    
    /* Typing indicator */
    .typing-indicator {
        color: #a0aec0;
        font-style: italic;
        margin: 10px 0;
    }
    
    @keyframes typing {
        0%, 20% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .typing-dots {
        animation: typing 1.5s infinite;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar with logo and options
    with st.sidebar:
        # Logo en el sidebar
        load_logo()
        st.markdown("---")
        
        st.header("🛠️ Options")
        
        # Session information
        st.info(f"**Session ID:** {st.session_state.session_id}")
        
        # Clear conversation button
        if st.button("🧹 Clear conversation", use_container_width=True, type="secondary"):
            clear_conversation()

    # Main content area - solo título sin logo
    st.markdown('<h1 class="main-header">Virtual Assistant</h1>', unsafe_allow_html=True)
    
    # Display message history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                with st.chat_message("user", avatar="👤"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(message["content"])

    # Input for new messages
    if prompt := st.chat_input("Type your question here..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Get response from backend with typing indicator
        with st.chat_message("assistant", avatar="🤖"):
            # Typing indicator
            typing_placeholder = st.empty()
            typing_placeholder.markdown('<div class="typing-indicator typing-dots">DraiverBot is typing...</div>', unsafe_allow_html=True)
            
            response = send_message_to_backend(prompt, st.session_state.session_id)
            
            # Clear typing indicator
            typing_placeholder.empty()
            
            if response:
                st.markdown(response)
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                error_msg = "I'm sorry, I couldn't process your message at this time. Please try again."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
