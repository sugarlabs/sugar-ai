# Running Sugar-AI with Docker

This project includes a [Dockerfile](Dockerfile) for containerized deployment. Follow these steps to build and run the Docker container:

1. **Build the Docker image:**  
   Open your terminal in the project's root directory and run:
   ```sh
   docker build -t sugar-ai .
   ```
2. **Run the Docker container:**
    If you have a GPU available (and are using the NVIDIA Docker runtime), run:
    ```sh
    docker run --gpus all -it --rm sugar-ai
    ```
    For a CPU-only system, simply run:
    ```sh
    docker run -it --rm sugar-ai
    ```

    The container will execute main.py on startup. Modify the Dockerfile if you need different entry behavior.

# Testing the FastAPI App

Install dependencies:
```sh
pip install -r requirements.txt
```

Run the server:
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test the endpoints:

-  A GET request to the root endpoint returns a welcome message:
- http://localhost:8000/
To ask a question, send a POST request to /ask with a query parameter question. For example:
```sh
curl -X POST "http://localhost:8000/ask?question=How%20do%20I%20create%20a%20Pygame%20window?"
```
The API will respond with a JSON object containing the answer.

### Additional Commands

#### CUDA Memory Issues:
If you encounter CUDA out-of-memory errors, try running on CPU by modifying the device parameter or setting:
```
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Running the RAG Agent in CLI mode:
You can also run the assistant directly from the command line. For example:
```sh
python rag_agent.py --quantize
```
Omit the --quantize flag if you prefer to run without 4‑bit quantization.
