import gradio as gr
import os
import zipfile
import requests
from huggingface_hub import hf_hub_download
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load embedding model
embed_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Download and load the Chroma database
def load_vector_db():
    chroma_db_path = hf_hub_download(
        repo_id="IJyad/NDMO_chroma_db",
        filename="chroma_db.zip",
        repo_type="dataset",
    )
    extract_dir = "./chroma_db"
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(chroma_db_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    return Chroma(persist_directory=extract_dir, embedding_function=embed_model)

vector_db = load_vector_db()

# Load API key from environment variables
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"

if not TOGETHER_API_KEY:
    raise ValueError("Missing Together AI API key. Set it in your environment.")

# Function to retrieve context from the vector database
def query_rag_system(question):
    best_doc = vector_db.similarity_search(question, k=1)[0]
    additional_docs = vector_db.similarity_search(question, k=3)
    additional_docs = [doc for doc in additional_docs if doc != best_doc]

    context = f"Best Chunk:\n{best_doc.page_content}\n\nAdditional Context:\n" + "\n".join(doc.page_content for doc in additional_docs)
    return context, best_doc, additional_docs

# Function to call Together AI for generating a response
def query_together_ai(prompt):
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "meta-llama/Llama-2-7b-chat-hf",
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7,
    }
    response = requests.post(TOGETHER_API_URL, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["text"]
    return "Error: Unable to generate a response from Together AI."

# Function to process chat interactions
def chat_interface(question, chat_history):
    if not question:
        return chat_history

    # Create a formatted history string from past interactions
    history_text = ""
    for user_msg, bot_msg in chat_history:
        if user_msg:
            history_text += f"User: {user_msg}\n"
        if bot_msg:
            history_text += f"Assistant: {bot_msg}\n"

    # Retrieve RAG context for the current question
    context, best_doc, additional_docs = query_rag_system(question)

    # Build the prompt including the conversation history and new question
    prompt = (
        f"{history_text}\n"
        f"Use the following context to answer the question concisely:\n{context}\n\n"
        f"Question: {question}"
    )

    # Get response from Together AI
    response_text = query_together_ai(prompt)
    sources = set(doc.metadata["category"] for doc in [best_doc] + additional_docs)
    final_response = f"{response_text}\n\n📌 **Sources**: {', '.join(sources)}"

    # Append messages with user on the right and chatbot on the left
    chat_history.append((question, None))  # User message
    chat_history.append((None, final_response))  # Chatbot response
    return chat_history

# Welcome message
welcome_message = """Hey! Welcome to the NDMO agent. Ask me anything related to NDMO regulations and compliance!"""

# Custom CSS for a light brown theme with a standard, full-width chat layout
custom_css = """
/* Global settings */
body {
  background-color: #FFF8F0;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  margin: 0;
  padding: 0;
}
/* Container for full-width content */
.container {
  max-width: 900px; /* Reduced width */
  width: 85%; /* Slightly smaller */
  margin: 0 auto;
  padding: 15px; /* Reduced padding */
  background-color: #FFF8F0;
}
/* Header styling */
.header {
  text-align: center;
  padding: 15px; /* Reduced padding */
  background-color: #C9A66B;
  color: #3E2723;
  border: 1px solid #A1887F;
  border-radius: 8px;
  margin-bottom: 15px; /* Reduced margin */
}
.header h1 {
  font-size: 24px; /* Slightly smaller */
  font-weight: bold;
  margin: 0;
  color: #3E2723;
}
.header p {
  font-size: 14px; /* Slightly smaller */
  margin: 6px 0 0;
  color: #5D4037;
}
/* Chat history container styling */
.chatbot-container {
  background-color: #FFF8F0;
  border: 1px solid #A1887F;
  border-radius: 8px;
  box-shadow: 0 3px 6px rgba(0,0,0,0.1);
  color: #3E2723;
  width: 90%; /* Reduced width */
  max-height: 250px; /* Reduced height */
  overflow-y: auto; /* Scroll if needed */
  padding: 8px; /* Reduced padding */
  font-size: 14px;
}
/* Input section styling */
.input-section {
  padding: 15px; /* Reduced padding */
  background-color: #FFF8F0;
  border-radius: 8px;
  color: #3E2723;
  display: flex;
  flex-direction: column;
}
/* Textbox styling */
.gr-textbox {
  border-radius: 15px; /* Reduced border radius */
  border: 1px solid #A1887F;
  padding: 10px; /* Reduced padding */
  font-size: 14px;
  background-color: #FFF8F0;
  color: #3E2723;
  width: 100%;
  min-height: 45px; /* Reduced height */
}
/* Button styling */
.gr-button {
  background-color: #C9A66B;
  color: #3E2723;
  border: none;
  border-radius: 6px; /* Smaller border radius */
  padding: 10px 15px; /* Reduced padding */
  font-size: 14px;
  cursor: pointer;
  transition: background-color 0.3s ease;
  margin-top: 8px; /* Reduced margin */
  width: 100%;
}
.gr-button:hover {
  background-color: #B38E5B;
}
"""

with gr.Blocks(css=custom_css) as demo:
    # Full-width container
    with gr.Column(elem_classes="container"):
        # Header using gr.HTML for custom HTML rendering
        gr.HTML(
            """
            <div class="header">
                <h1>NDMO Regulations Agent</h1>
                <p>Your trusted assistant for NDMO regulations and compliance. Ask me anything!</p>
            </div>
            """
        )

        # Chat history container
        chatbot = gr.Chatbot(label="Chat History", elem_classes="chatbot-container")

        # Input section grouping input components
        with gr.Column(elem_classes="input-section"):
            question = gr.Textbox(
                lines=3,
                placeholder="Type your question here...",
                label="Your Question",
                elem_classes="gr-textbox"
            )
            submit_button = gr.Button("Send", elem_classes="gr-button")

        # State to maintain conversation history
        chat_history = gr.State([])

        # Define interaction: update chat history when the button is clicked
        submit_button.click(
            fn=chat_interface,
            inputs=[question, chat_history],
            outputs=chatbot,
        )

demo.launch(share=True)
demo.launch(share=True)