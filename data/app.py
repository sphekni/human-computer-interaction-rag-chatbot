import gradio as gr
from rag import RAGChatbot

bot = RAGChatbot()

def chat(user_input):
    if not user_input or not user_input.strip():
        return "Please enter a question or message."
    return bot.generate(user_input)

ui = gr.Interface(
    fn=chat,
    inputs=gr.Textbox(label="Ask me anything about HCI / RAG", lines=3),
    outputs=gr.Textbox(label="Chatbot Response", lines=6),
    title="HCI RAG Chatbot",
    description="A simple Retrieval-Augmented Generation chatbot built for a Human-Computer Interaction group project.",
)

if __name__ == "__main__":
    ui.launch()