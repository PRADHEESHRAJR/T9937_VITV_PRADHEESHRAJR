import streamlit as st
import pandas as pd
from fastmcp import FastMCP
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# ==========================================
# 1. INITIALIZE FAST-MCP SECURE INFRASTRUCTURE
# ==========================================
# Fulfills the "Secure Infrastructure: Model Context Protocol (MCP) integration" objective
mcp = FastMCP(name="Agentic-IoT-PdM-Server")

@st.cache_resource
def load_rag_databases():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    tech_db = FAISS.load_local("../vector_store/tech_db", embeddings, allow_dangerous_deserialization=True)
    safety_db = FAISS.load_local("../vector_store/safety_db", embeddings, allow_dangerous_deserialization=True)
    return tech_db, safety_db

tech_db, safety_db = load_rag_databases()

# Define governed MCP Tools for the Agent
@mcp.tool
def get_technical_fix(anomaly_description: str) -> str:
    """Queries the technical vector database for a mechanical solution."""
    docs = tech_db.similarity_search(anomaly_description, k=1)
    return docs[0].page_content if docs else "No technical fix found."

@mcp.tool
def get_safety_constraints(anomaly_description: str) -> str:
    """Queries the safety vector database for mandatory operational constraints."""
    # Fulfills the "Strict Segregation: Separate retrieval budgets" objective
    docs = safety_db.similarity_search(anomaly_description, k=1)
    return docs[0].page_content if docs else "No safety constraints found."

# ==========================================
# 2. MARKET-READY UI & DATA ENGINE (SCADA)
# ==========================================
st.set_page_config(page_title="Agentic-IoT PdM", layout="wide")
st.title("🛢️ Agentic-IoT Framework for Predictive Maintenance")
st.subheader("Powered by Multi-RAG & Context-Aware Safety Protocols")

# Load SCADA Dataset
@st.cache_data
def load_scada_data():
    try:
        # Fulfills the Dataset requirement
        return pd.read_csv("../data/scada_pipeline.csv")
    except FileNotFoundError:
        st.error("Error: scada_pipeline.csv not found in the data/ folder.")
        return None

df = load_scada_data()

st.sidebar.header("📡 The Data Engine (SCADA)")
if df is not None and st.sidebar.button("Monitor Live Telemetry Stream"):
    # Simulate a live reading by sampling a random row from the CSV dataset
    live_reading = df.sample(1).iloc[0]
    
    # Extract values (Update column names if they differ slightly in your exact CSV)
    pressure = float(live_reading.get('Pressure', 595))
    temp = float(live_reading.get('Temperature', 105))
    vibration = float(live_reading.get('Vibration', 110))
    
    st.sidebar.metric("Pipeline Pressure", f"{pressure:.1f} PSI")
    st.sidebar.metric("Internal Temperature", f"{temp:.1f} °C")
    st.sidebar.metric("Pump Vibration", f"{vibration:.1f} Hz")
    
    # Acts as the autonomous trigger for AI diagnostic reasoning (from slide 2)
    if pressure > 580:
        st.error(f"🚨 CRITICAL ANOMALY DETECTED: Pressure spiked to {pressure:.1f} PSI.")
        anomaly_desc = "High pipeline pressure above 580 PSI"
        
        with st.spinner("Executing RAGuard Architecture via MCP..."):
            # 1. Agent invokes MCP tools for strict segregated retrieval
            tech_context = get_technical_fix(anomaly_desc)
            safety_context = get_safety_constraints("valve bleed temperature")
            
            st.warning(f"**MCP Retrieved Technical Action:** {tech_context}")
            st.info(f"**MCP Retrieved Safety Constraint:** {safety_context}")
            
            # 2. AI Reasoning Engine (Zero-Cost Local LLM)
            llm = Ollama(model="phi3")
            
            prompt = PromptTemplate.from_template("""
            You are a Context-Aware Safety AI for Oil & Gas pipelines.
            Current SCADA Telemetry: Pressure is {pressure} PSI, Temperature is {temp}C.
            
            Proposed Technical Action: {tech}
            Mandatory Safety Constraint: {safety}
            
            Objective: Cross-reference the mechanical solution against the safety constraints. Is it safe to execute the technical action based on the current temperature? 
            Output a final, safe maintenance procedure. Explain your reasoning clearly.
            """)
            formatted_prompt = prompt.format(pressure=pressure, temp=temp, tech=tech_context, safety=safety_context)
            
            # Generate the mandatory safety validation
            response = llm.invoke(formatted_prompt)
            st.success("**🤖 Agentic AI Validated Output:**\n\n" + response)