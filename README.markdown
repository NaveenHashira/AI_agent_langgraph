# 🤖 AI Agent with LangGraph, LangChain, and Streamlit 🚀

This project implements an AI agent using [LangGraph](https://github.com/langchain-ai/langgraph) 🌐 and [LangChain](https://github.com/langchain-ai/langchain) ⛓️, integrated with a [Streamlit](https://streamlit.io/) 📊 app for user-friendly interaction. The agent handles mathematical operations (e.g., addition, multiplication) ➕✖️ and retrieves information from external sources like Wikipedia, Arxiv, and DuckDuckGo 🔍. It provides concise answers and maintains conversation memory within a session 🧠.

## ⚙️ Installation

1. Ensure you have [Python](https://www.python.org/downloads/) 🐍 installed (version 3.8 or higher recommended).
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required libraries:
   ```bash
   pip install langgraph langchain-core langchain-groq langchain-community streamlit python-dotenv
   ```
4. Set up your GROQ_API_KEY 🔑. Create a `.env` file in the project directory with:
   ```
   GROQ_API_KEY=your_api_key_here
   ```
   Replace `your_api_key_here` with your actual [GROQ API key](https://console.groq.com/).

## 🚀 Usage

### Running the Streamlit App 📱
1. Ensure the `.env` file is configured with your GROQ_API_KEY 🔑
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Open your browser 🌐 and navigate to `http://localhost:8501`
4. Enter a query in the text input field and click "Submit" to receive the final answer 💬

### Running the AI Agent Directly 💻
You can use the AI agent programmatically by importing the `run_agent` function from `ai_agent.py`. Example:
```python
from ai_agent import run_agent

thread_id = "my_thread"
answer = run_agent("What is 5 + 3?", thread_id=thread_id)
print(answer["messages"][-1].content)  # Outputs: "8" ✅
```

## 💡 Examples

| Query | Answer |
|-------|--------|
| What is 5 + 3? ➕ | 8 |
| What is the capital of France? 🗼 | Paris |
| What was the enrollment count of the clinical trial on H. pylori from Jan-May 2018? 🧪 | 90 |

These examples demonstrate the agent's ability to handle mathematical queries ➗ and retrieve information from external sources 🔍.

## 🛠️ How It Works

The AI agent is built using LangGraph 🌐, enabling stateful workflows with **qwen-qwq-32b** 🤖 as the language model. It supports:

- **🧮 Mathematical Tools**: Addition, subtraction, multiplication, division, modulus
- **🔍 Knowledge Retrieval**: Wikipedia, Arxiv, and DuckDuckGo search
- **🧠 Memory**: Conversation history is maintained within a session using LangGraph's MemorySaver
- **💬 Concise Responses**: A system prompt ensures only the final answer is returned, without reasoning or extra text

The Streamlit app 📊 provides a simple interface for entering queries and viewing answers.

## 📂 Project Structure
- `ai_agent.py` 🤖: Defines the AI agent, tools, and LangGraph workflow
- `app.py` 📱: Streamlit app for user interaction
- `.env` 🔑: Stores the GROQ_API_KEY
- `README.md` 📄: This documentation

## 👥 Contributing
Contributions are welcome! 🙌 Please open an issue or submit a pull request on [GitHub](https://github.com/) for suggestions or improvements.

## ⚠️ Notes
- 🌐 An internet connection is required for external API calls
- 🔑 Ensure your GROQ_API_KEY is valid to avoid authentication errors
- ⚡ Powered by GROQ's high-performance LLM inference

## ✨ Key Features
- **🧠 Stateful Conversations**: Maintains context across interactions
- **⚡ Blazing Fast**: Leverages GROQ's hardware acceleration
- **🔧 Extensible Architecture**: Easily add new tools and capabilities
- **📱 User-Friendly Interface**: Streamlit makes interaction intuitive

