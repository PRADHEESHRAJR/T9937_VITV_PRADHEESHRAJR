import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# ==========================================
# 1. MARKET-READY UI SETUP
# ==========================================
st.set_page_config(page_title="Agentic-FL Control Plane", layout="wide")
st.title("🌐 Agentic-FL System for Retail Demand Forecasting")
st.subheader("Autonomous Privacy-Preserving Federated Orchestrator")

# ==========================================
# 2. DATA PROCESSING & EDGE MACHINE LEARNING
# ==========================================
@st.cache_data
def load_and_partition_data():
    """Reads the Rossmann dataset and splits it to simulate heterogeneous edge stores."""
    try:
        df = pd.read_csv("../data/train.csv", low_memory=False)
        
        # Isolate two stores to represent the unpredictable, non-IID edge environments
        store_a = df[df["Store"] == 1].copy().dropna(subset=["Customers", "Promo", "Sales"])
        store_b = df[df["Store"] == 2].copy().dropna(subset=["Customers", "Promo", "Sales"])
        
        return store_a, store_b
    except FileNotFoundError:
        st.error("Error: train.csv not found in the data/ folder. Please add the Rossmann dataset.")
        return None, None

def train_edge_model(data, store_name):
    """Simulates distributed Scikit-Learn training on edge network hardware."""
    X = data[["Customers", "Promo"]]
    y = data["Sales"]
    
    # Scale data for the linear regressor
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
    model.fit(X_scaled, y)
    
    # Calculate local loss metric (Mean Squared Error approximation)
    predictions = model.predict(X_scaled)
    mse = float(np.mean((predictions - y) ** 2))
    
    metadata = {
        "store": store_name,
        "data_volume": len(data),
        "loss_mse": round(mse, 2),
        "coefficients": model.coef_.tolist(),
        "intercept": model.intercept_.tolist()
    }
    return metadata

# ==========================================
# 3. AGENTIC AI ORCHESTRATION ENGINE
# ==========================================
store_a_data, store_b_data = load_and_partition_data()

if store_a_data is not None and store_b_data is not None:
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Edge Node A (Store 1)**\n- Local Data Volume: {len(store_a_data)} records")
    with col2:
        st.info(f"**Edge Node B (Store 2)**\n- Local Data Volume: {len(store_b_data)} records")
        
    if st.button("🚀 Initiate Agentic Federated Orchestration", type="primary"):
        
        with st.spinner("Training local edge models (preserving data privacy)..."):
            meta_a = train_edge_model(store_a_data, "Store A")
            meta_b = train_edge_model(store_b_data, "Store B")
            
        st.success("Edge training complete. Metadata extracted successfully.")
        
        with st.expander("View Real-Time Edge Node Metadata"):
            st.json({"Store A Metadata": meta_a, "Store B Metadata": meta_b})

        with st.spinner("Agentic AI evaluating metadata and adapting mathematical aggregation..."):
            # Initialize Zero-Cost Local LLM
            llm = Ollama(model="phi3", temperature=0.0) 
            
            prompt = PromptTemplate.from_template("""
            You are an Agentic AI Orchestrator acting as a mathematical control plane for a Federated Learning network.
            Your objective is to mitigate systemic bias by assigning aggregation weights to two retail edge models based on their performance.
            
            Store A Metadata -> Loss MSE: {loss_a} | Data Volume: {vol_a}
            Store B Metadata -> Loss MSE: {loss_b} | Data Volume: {vol_b}
            
            Rules for Adaptive Aggregation:
            1. The store with the LOWER Loss MSE is more accurate and must receive a higher weight (e.g., 0.6 to 0.8).
            2. The store with the HIGHER Loss MSE must receive a lower weight.
            3. The two weights MUST sum exactly to 1.0.
            
            Respond strictly with a valid JSON object containing keys "weight_a" and "weight_b". Do not include Markdown, backticks, or any other text.
            """)
            
            formatted_prompt = prompt.format(
                loss_a=meta_a["loss_mse"], vol_a=meta_a["data_volume"],
                loss_b=meta_b["loss_mse"], vol_b=meta_b["data_volume"]
            )
            
            # LLM autonomously dictates the aggregation strategy
            response = llm.invoke(formatted_prompt)
            
            try:
                # Clean response to parse JSON
                clean_response = response.replace("```json", "").replace("```", "").strip()
                weights = json.loads(clean_response)
                
                st.subheader("🧠 Agentic AI Output & Aggregation")
                st.markdown("The LLM has successfully transcended chatbot roles to act as an intelligent control plane.")
                
                st.metric("Store A Assigned Weight", f"{weights['weight_a']}x")
                st.metric("Store B Assigned Weight", f"{weights['weight_b']}x")
                
                # Apply the Agent's mathematical strategy
                global_coef = [
                    (meta_a["coefficients"][0] * weights['weight_a']) + (meta_b["coefficients"][0] * weights['weight_b']),
                    (meta_a["coefficients"][1] * weights['weight_a']) + (meta_b["coefficients"][1] * weights['weight_b'])
                ]
                
                st.success(f"**Maximized Global Forecasting Model Coefficients:** {global_coef}")
                st.caption("Data privacy preserved: Raw customer data never left the local edge nodes.")
                
            except Exception as e:
                st.error("The Agentic AI failed to return a strict mathematical JSON object. Please run again.")
                st.code(response)