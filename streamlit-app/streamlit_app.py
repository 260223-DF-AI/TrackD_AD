# run this with: streamlit run streamlit-app/streamlit_app.py
import streamlit as st
from app_functionality import LLM_analysis, LLM_analysis_TOT, prediction
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
    set_bg("https://imgs.search.brave.com/b5W11UKcxmWxqLRjOVAJbd52zLWozhTBuLThKigsBNQ/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9jZG4u/dmVjdG9yc3RvY2su/Y29tL2kvcHJldmll/dy0xeC8yOS85NC9o/b3VzZS1yZWFsdHkt/Z29sZC1sb2dvLXZl/Y3Rvci00MTI5Mjk5/NC5qcGc")
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
    set_bg("https://imgs.search.brave.com/b5W11UKcxmWxqLRjOVAJbd52zLWozhTBuLThKigsBNQ/rs:fit:500:0:1:0/g:ce/aHR0cHM6Ly9jZG4u/dmVjdG9yc3RvY2su/Y29tL2kvcHJldmll/dy0xeC8yOS85NC9o/b3VzZS1yZWFsdHkt/Z29sZC1sb2dvLXZl/Y3Rvci00MTI5Mjk5/NC5qcGc")
    st.header("LLM Analysis")
    st.write("Upload an image of a room in a house to see an analysis of the image based on the model's predictions, along with pricing strategies and a sales pitch for the property.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image')
        if st.button("Get LLM Analysis (Chain of Thought)"):
            # Call the API endpoint with the uploaded image and display the results
            result = LLM_analysis(uploaded_file)
            st.write("LLM Analysis:")
            st.write(result)
        if st.button("Get LLM Analysis (Tree of Thought)"):
            # Call the API endpoint with the uploaded image and display the results
            result = LLM_analysis_TOT(uploaded_file)
            st.write("LLM Analysis:")
            st.write(result)
    if st.button("Home"):
        st.session_state.page = "home"


# Session state tracker
if st.session_state.page == "home":
    home()
elif st.session_state.page == "model_predictions":
    model_setup()
elif st.session_state.page == "LLM_analysis":
    LLM()
