async def run_openai(input_text, config):
    """
    OpenAI provider implementation (asynchronous).
    Requires OPENAI_API_KEY to be set in environment.
    """
    import openai
    import os
    
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not found in environment"}
        
    client = openai.AsyncOpenAI(api_key=api_key)
    
    try:
        response = await client.chat.completions.create(
            model=config.get("model", "gpt-3.5-turbo"),
            messages=[{"role": "user", "content": input_text}]
        )
        return {"response": response.choices[0].message.content}
    except Exception as e:
        return {"error": str(e)}
