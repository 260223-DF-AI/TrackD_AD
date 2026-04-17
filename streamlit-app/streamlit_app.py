# run this with: streamlit run streamlit-app/streamlit_app.py
import streamlit as st
import json
import re
from app_functionality import LLM_analysis, LLM_analysis_TOT, prediction

# Theme colors (matches .streamlit/config.toml)
_THEME_TEXT_COLOR = "white"
_THEME_LINK_COLOR = "darkOrchid"

# Inject CSS to ensure link color and text color apply inside the app
_THEME_CSS = f"""
<style>
    .stApp, .stApp * {{ color: {_THEME_TEXT_COLOR} !important; }}
    a, a * {{ color: {_THEME_LINK_COLOR} !important; }}
    .stMarkdown {{ color: {_THEME_TEXT_COLOR} !important; }}
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)

st.title("Estate Insight")

# Set background image
def set_bg(img_url):
    css = f"""
    <style>
    .stApp {{
        background-image: url({img_url});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def _display_llm_result(result):
    """Render JSON-like results as readable markdown text.

    This will walk dict/list structures and render keys as bold
    headings with their values as paragraphs. Plain strings are
    displayed as paragraphs. Falls back to `st.json` only for
    non-serializable complex types.
    """

    def _render(obj, key_name=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                header = f"**{k}:**"
                st.markdown(header)
                _render(v, key_name=k)
        elif isinstance(obj, list):
            # If list of simple strings, join into paragraphs
            if all(isinstance(x, (str, int, float)) for x in obj):
                for item in obj:
                    text = str(item).strip()
                    if text:
                        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
                        for p in paragraphs:
                            p = re.sub(r"\n+", " ", p).strip()
                            st.markdown(p)
                        st.markdown("\n")
            else:
                # Mixed list: render each element recursively with a bullet
                for i, item in enumerate(obj, start=1):
                    st.markdown(f"- Item {i}")
                    _render(item)
        elif isinstance(obj, (str, int, float)):
            text = str(obj).strip()
            if not text:
                return
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            for p in paragraphs:
                p = re.sub(r"\n+", " ", p).strip()
                st.markdown(p)
            st.markdown("\n")
        else:
            # Fallback to JSON display for unknown types
            try:
                st.json(obj)
            except Exception:
                st.write(str(obj))

    _render(result)

# session state to monitor what page the user is on
if "page" not in st.session_state:
    st.session_state.page = "home"

# Home page with descriptions of resources and buttons to navigate to different pages
def home():
    set_bg("https://imgs.search.brave.com/b5W11UKcxmWxqLRjOVAJbd52zLWozhTBuLThKigsBNQ/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9jZG4u/dmVjdG9yc3RvY2su/Y29tL2kvcHJldmll/dy0xeC8yOS85NC9o/b3VzZS1yZWFsdHkt/Z29sZC1sb2dvLXZl/Y3Rvci00MTI5Mjk5/NC5qcGc")
    st.header("Available Resources")
    st.subheader("Model Predictions")
    st.write("View the predicted features and quality of a room in an image, along with a sales pitch for the property based on the model's predictions.")
    if st.button("Go to Model Predictions"):
        st.session_state.page = "model_predictions"
    if st.button("Go to LLM Analysis"):
        st.session_state.page = "LLM_analysis"

def model_setup():
    # set_bg("https://imgs.search.brave.com/b5W11UKcxmWxqLRjOVAJbd52zLWozhTBuLThKigsBNQ/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9jZG4u/dmVjdG9yc3RvY2su/Y29tL2kvcHJldmll/dy0xeC8yOS85NC9o/b3VzZS1yZWFsdHkt/Z29sZC1sb2dvLXZl/Y3Rvci00MTI5Mjk5/NC5qcGc")
    st.header("Model Predictions")
    st.write("Upload an image of a room in a house to see the model's predicted features and quality of the room.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image')
        if st.button("Get Model Prediction"):
            # Call the API endpoint with the uploaded image and display the results
            result = prediction(uploaded_file)
            st.write("Model Prediction:")
            st.write(result)
    if st.button("Home"):
        st.session_state.page = "home"

def LLM():
    # set_bg("https://imgs.search.brave.com/b5W11UKcxmWxqLRjOVAJbd52zLWozhTBuLThKigsBNQ/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9jZG4u/dmVjdG9yc3RvY2su/Y29tL2kvcHJldmll/dy0xeC8yOS85NC9o/b3VzZS1yZWFsdHkt/Z29sZC1sb2dvLXZl/Y3Rvci00MTI5Mjk5/NC5qcGc")
    st.header("LLM Analysis")
    st.write("Upload an image of a room in a house to see an analysis of the image based on the model's predictions, along with pricing strategies and a sales pitch for the property.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image')
        if st.button("Get LLM Analysis (Chain of Thought)"):
            # Call the API endpoint with the uploaded image and display the results
            result = LLM_analysis(uploaded_file)
            st.write("LLM Analysis:")
            _display_llm_result(result)
        if st.button("Get LLM Analysis (Tree of Thought)"):
            # Call the API endpoint with the uploaded image and display the results
            result = LLM_analysis_TOT(uploaded_file)
            st.write("LLM Analysis:")
            _display_llm_result(result)
    if st.button("Home"):
        st.session_state.page = "home"


# Session state tracker
if st.session_state.page == "home":
    home()
elif st.session_state.page == "model_predictions":
    model_setup()
elif st.session_state.page == "LLM_analysis":
    LLM()
