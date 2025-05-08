from openai import OpenAI
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
from openai import OpenAI
import torch

# --- Setup and Configuration ---
st.title("🧠 Response Evaluation Agent")

# Load embedding model (handle potential errors gracefully)
@st.cache_resource
def load_embedding_model():
    try:
        # Explicitly load the model to the CPU
        return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    except Exception as e:
        st.error(f"Error loading SentenceTransformer: {e}")
        return None

embedding_model = load_embedding_model()

# Retrieve Mistral API key and base URL from Streamlit secrets
mistral_api_key = st.secrets.get("mistral", {}).get("api_key")
mistral_api_base_url = st.secrets.get("mistral", {}).get("base_url", "https://api.mistral.ai/v1")

if not mistral_api_key:
    st.error("Mistral API key not found in Streamlit secrets. Please add it to .streamlit/secrets.toml (or Streamlit Cloud).")

# Initialize the OpenAI client with Mistral credentials (only if API key is available)
@st.cache_resource(show_spinner=False)
def create_openai_client(api_key, base_url):
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url)
    return None

client = create_openai_client(mistral_api_key, mistral_api_base_url)

# --- Helper Functions ---
def get_cosine_similarity(source, response, model):
    """Compute cosine similarity between source and response text."""
    if model is None:
        return 0.0
    emb1 = model.encode(source, convert_to_tensor=True)
    emb2 = model.encode(response, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(emb1, emb2).item()
    return round(cosine_sim * 100, 2)

def evaluate_with_mistral(source, response, openai_client):
    """Evaluate the response based on the source using Mistral."""
    if not openai_client:
        return {"score": 0, "explanation": "Mistral API key is missing. Cannot evaluate."}

    prompt = f"""
You are an evaluation agent.

Source Text: "{source}"
Response Text: "{response}"

Evaluate how well the response captures the meaning and important points of the source.
Give a score from 0 to 100.
If the response is an *exact* copy of the source text, the score should be 100.
If there are any points missing from the response, list them.
If the meaning is incorrect or reversed, state that clearly.
Finally, provide a short explanation.

Return your answer as JSON in the format:
{{
  "score": <number from 0 to 100>,
  "explanation": "<explanation of the score>"
}}
"""

    try:
        chat_response = openai_client.chat.completions.create(
            model="mistral-tiny",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        content = chat_response.choices[0].message.content.strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if "Score:" in content:
                try:
                    score = int(content.split("Score:")[1].strip())
                    return {"score": score, "explanation": "Evaluation failed to provide a full explanation."}
                except ValueError:
                    return {"score": 0, "explanation": f"Invalid response format: {content}"}
            else:
                return {"score": 0, "explanation": f"Invalid response format: {content}"}
    except openai.OpenAIError as e:
        return {"score": 0, "explanation": f"Mistral API Error: {str(e)}"}
    except Exception as e:
        return {"score": 0, "explanation": f"Error during evaluation: {str(e)}"}

# --- Streamlit UI Elements ---
# Predefined source text
default_source_text = """The Indus Water Treaty (IWT) is a water-distribution treaty between India and Pakistan, arranged and negotiated by the World Bank, to use the water available in the Indus River and its tributaries. It was signed in Karachi on 19 September 1960 by Indian prime minister Jawaharlal Nehru and Pakistani president Ayub Khan. On 23 April 2025, the Government of India suspended the treaty, citing national security concerns and alleging Pakistan’s support of state-sponsored terrorism."""

# Text input fields
source_text = st.text_area("✏️ Source Text (Rubric)", max_chars=800, height=200, value=default_source_text)
response_text = st.text_area("📝 Response Text", max_chars=800, height=150, placeholder="Summarize the Indus Water Treaty from the information provided in the Rubric above")

# Evaluation button
if st.button("🔍 Evaluate"):
    if not source_text.strip() or not response_text.strip():
        st.warning("Please enter both source and response text.")
    elif len(source_text.split()) > 200 or len(response_text.split()) > 200:
        st.warning("Please limit both source and response text to a reasonable number of words.")
    elif embedding_model is None:
        st.error("Embedding model failed to load. Please check your environment.")
    elif client is None:
        st.error("Mistral API client could not be initialized. Please check your API key in Streamlit secrets.")
    else:
        with st.spinner("Evaluating..."):
            # Calculate cosine similarity
            cosine_score = get_cosine_similarity(source_text, response_text, embedding_model)
            # Evaluate with Mistral
            mistral_eval = evaluate_with_mistral(source_text, response_text, client)
            # Combine scores
            final_score = round((0.7 * mistral_eval["score"]) + (0.3 * cosine_score), 2)

        # Display results
        st.subheader(f"✅ Final Score: {final_score}%")
        st.markdown("---")
        st.subheader("🧾 Explanation")
        st.write(mistral_eval["explanation"])