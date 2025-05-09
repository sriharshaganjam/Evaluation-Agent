from openai import OpenAI
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import json
import torch
import re  # Import the regular expression library

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

    if len(response.split()) < 10:
        return {"score": 0, "explanation": "The provided answer is too short and does not adequately explain the source. Please provide a more detailed response."}

    prompt = f"""
You are a strict evaluation agent. Your sole task is to evaluate the "Response Text" based on the "Source Text" provided below. You MUST adhere to the following instructions and return your answer strictly in the JSON format specified.

Source Text: "{source}"
Response Text: "{response}"

Evaluation Criteria:
1. Accuracy: How well does the response accurately capture the key information from the source text?
2. Completeness: Does the response cover all the important points from the source text?
3. Conciseness: Is the response concise and avoids unnecessary information?
4. Originality: Is the response an original summary or is it an exact copy of the source?

Scoring: Provide a score from 0 to 100.

Specific Instructions:
- If the response is an EXACT copy of the source text, the score MUST be 0, and the explanation MUST be: "The response is an exact copy of the source. Please provide an original summary."
- Identify any key points from the source text that are missing in the response and include them in the "explanation".
- If the response contains incorrect or misleading information, clearly state this in the "explanation" and deduct points from the score accordingly.
- The "explanation" should be a concise summary of your evaluation, justifying the assigned score and suggesting areas for improvement.

Return your evaluation as a JSON object with the following structure:
{{
  "score": <integer between 0 and 100>,
  "explanation": "<string containing your evaluation>"
}}

Do not include any conversational elements, creative writing, or information outside the scope of evaluating the provided response against the source text. Your response MUST be valid JSON.
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
            evaluation = json.loads(content)
            return evaluation
        except json.JSONDecodeError:
            if "Score:" in content:
                try:
                    match = re.search(r"Score:\s*(\d+)", content)
                    if match:
                        score = int(match.group(1))
                        explanation = content.split("Score:")[0].strip()
                        return {"score": score, "explanation": explanation}
                    else:
                        return {"score": 0, "explanation": f"Invalid response format: {content}"}
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