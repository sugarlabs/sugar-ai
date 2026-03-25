async def run_model(input_text, provider="openai", config=None):
    """
    Modular asynchronous router for multiple AI providers.
    Supports: openai, huggingface, local
    """
    try:
        if provider == "openai":
            from sugar_ai.providers.openai_provider import run_openai
            return await run_openai(input_text, config)

        elif provider == "huggingface":
            from sugar_ai.providers.huggingface_provider import run_hf
            return await run_hf(input_text, config)

        elif provider == "local":
            from sugar_ai.providers.local_provider import run_local
            return await run_local(input_text, config)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    except Exception as e:
        import logging
        logger = logging.getLogger("sugar-ai")
        logger.error(f"Error in model router: {str(e)}")
        return {"error": str(e)}
