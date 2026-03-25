async def run_hf(input_text, config):
    """
    HuggingFace provider implementation (asynchronous signature).
    Uses 'transformers' pipeline.
    """
    from transformers import pipeline
    
    model_name = config.get("model", "gpt2")
    try:
        # In production this might be offloaded to an executor to avoid blocking the loop
        pipe = pipeline("text-generation", model=model_name)
        result = pipe(input_text, max_length=config.get("max_length", 200))
        return {"response": result[0]["generated_text"]}
    except Exception as e:
        return {"error": str(e)}
