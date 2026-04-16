import requests
API_BASE_URL = "http://localhost:8000"

def prediction(image):
    try:
        files = {"image": (image.name, image, image.type)}
        resp = requests.post(f"{API_BASE_URL}/analyze", files=files, timeout=100)
        print(resp)
        # result = resp.json()
        return resp.json()
    except Exception as e:
        return (str(e))
    
def LLM_analysis(image):
    try:
        files = {"image": (image.name, image, image.type)}
        resp = requests.post(f"{API_BASE_URL}/bedrock", files=files, timeout=100)
        print(resp)
        # result = resp.json()
        return resp.json()
    except Exception as e:
        return (str(e))
    
def LLM_analysis_TOT(image):
    try:
        files = {"image": (image.name, image, image.type)}
        resp = requests.post(f"{API_BASE_URL}/bedrock/TreeOfThoughtPrompt", files=files, timeout=100)
        print(resp)
        # result = resp.json()
        return resp.json()
    except Exception as e:
        return (str(e))