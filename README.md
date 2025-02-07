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
