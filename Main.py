NEO4J_URI="neo4j+s://7ce2ba95.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="7rpHkvW_zsfPDPt6GftaxrHkHMoouOb22gPWTHkKezs"
import os
os.environ["NEO4J_URI"]=NEO4J_URI
os.environ["NEO4J_USERNAME"]=NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"]=NEO4J_PASSWORD
from langchain_community.graphs import Neo4jGraph
graph=Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)
groq_api_key="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"
from langchain_groq import ChatGroq

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.3-70b-versatile")
import streamlit as st
import pdfplumber
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from py2neo import Graph

# Load and process the PDF
pdf_path = "Harry_Potter.pdf"
chunk_size = 2  # Process pages in chunks
documents = []

with pdfplumber.open(pdf_path) as pdf:
    total_pages = len(pdf.pages)
    for i in range(0, total_pages, chunk_size):
        chunk_text = ""
        for j in range(i, min(i + chunk_size, total_pages)):
            chunk_text += pdf.pages[j].extract_text()
        documents.append(Document(page_content=chunk_text))

# Initialize LLMGraphTransformer
llm = ChatOpenAI(model_name="gpt-4", temperature=0)  # Use GPT-4 for better accuracy
llm_transformer = LLMGraphTransformer(llm=llm)

# Convert extracted text into graph structure
graph_documents = llm_transformer.convert_to_graph_documents(documents)



# Initialize GraphCypherQAChain
qa_chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True)

# Streamlit UI
st.title("📖 Harry Potter Chatbot 🧙‍♂️")
st.write("Ask anything about the Harry Potter universe!")

# User Input
user_query = st.text_input("Type your question here:")

if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking..."):
            response = qa_chain.run(user_query)
            st.markdown(f"**Chatbot:** {response}")
    else:
        st.warning("Please enter a question!")

# Optional: Display Graph Data (for debugging)
if st.checkbox("Show Graph Data"):
    nodes = graph.run("MATCH (n) RETURN n LIMIT 5").data()
    relationships = graph.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 5").data()
    st.write("**Sample Nodes:**", nodes)
    st.write("**Sample Relationships:**", relationships)
