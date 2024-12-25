import pytest
import streamlit as st
from datetime import datetime

@pytest.mark.unit
def test_session_state_initialization():
    if 'processed_text' not in st.session_state:
        st.session_state.processed_text = None
    if 'chunks' not in st.session_state:
        st.session_state.chunks = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    assert st.session_state.processed_text is None
    assert st.session_state.chunks is None
    assert st.session_state.messages == []

@pytest.mark.unit
def test_message_history():
    st.session_state.messages = []
    st.session_state.messages.append({
        "content": "test message",
        "timestamp": datetime.now(),
        "role": "user"
    })
    
    assert len(st.session_state.messages) == 1
    assert st.session_state.messages[0]["content"] == "test message"