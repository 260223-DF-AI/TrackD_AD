# TrackD_AD

## Datasets from specific locations
- Housing room images with labels on the quality of the room: https://huggingface.co/datasets/sk2003/houzz-data/viewer
- Dataset with many images of different places: https://www.kaggle.com/datasets/mittalshubham/images256
- Retrieved inspiration and some of the datasets from this article: https://www.sciencedirect.com/science/article/pii/S2667305322000217#sec0002

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