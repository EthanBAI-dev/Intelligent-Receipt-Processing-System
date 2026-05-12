# 1. Import the library
from inference_sdk import InferenceHTTPClient

# 2. Connect to your workflow
client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="zvFWvtuVgoJHzpG0tFp0"
)

# 3. Run your workflow on an image
result = client.run_workflow(
    workspace_name="ras-workspace-uhnsw",
    workflow_id="custom-workflow",
    images={
        "image": "/Users/baiwenbin/Documents/MyProjects/DeepLearning Project/work/processed data2/IMG_3134.jpg" # Path to your image file
    },
    use_cache=True # Speeds up repeated requests
)

# 4. Get your results
print(result)

