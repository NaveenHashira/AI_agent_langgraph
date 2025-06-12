import streamlit as st
import uuid
from ai_agent import run_agent  

# Set up Streamlit app
st.title("AI Agent App ")

# Initialize session state
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # Unique thread_id per session

# User input
query = st.text_input("Enter your query:")

# Submit button
if st.button("Submit"):
    if query:
        # Call run_agent with query and thread_id
        result = run_agent(query=query, thread_id=st.session_state.thread_id)
        
        # Extract final answer from result
        final_answer = result["messages"][-1].content
        
        # Display only the final answer
        st.write(final_answer)