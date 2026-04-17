# TrackD_AD

## Track D: EstateInsight (Property Valuer & Agent)

- **The Problem**: A real-estate firm needs to scale their property quality assessments.
- **Vision Task**: Classify room types and assessment quality (e.g., `Kitchen-Renovated`, `Bathroom-Original`, `Exterior-Basic`).
- **Reasoning Task**: Use **Tree of Thought (ToT)** to explore pricing strategies and **ReAct** to generate a "Sales Pitch" based on the photo quality and neighborhood description.

## Datasets from specific locations
- Housing room images with labels on the quality of the room: https://huggingface.co/datasets/sk2003/houzz-data/viewer
- Dataset with many images of different places: https://www.kaggle.com/datasets/mittalshubham/images256
- Random houses from Zillow
- Retrieved inspiration and some of the datasets from this article: https://www.sciencedirect.com/science/article/pii/S2667305322000217#sec0002

## Installation

Use the Python package manager [pip](https://pip.pypa.io/en/stable/) to install the project dependencies.

```bash
pip install -r requirements.txt
```

## Running Instructions (from terminal)
### navigate to ~/TrackD_AD
**Uvicorn:** (FastAPI)
    
    uvicorn app.main:app --reload

**Tensorboard:** (Tracking training metrics)

    To view the dashboard, run this in a terminal:
    tensorboard --logdir=./runs

**Streamlit:** (Interactive App)

    streamlit run streamlit-app/streamlit_app.py



## Contributors
- Alec Figueroa
- Dio Soetarman