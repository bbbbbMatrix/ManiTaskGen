# Calling VLM (Vision Language Models)

To call VLM, we cloned VLMEvalKit in the early development stage and implemented the functionality to call OpenRouter API based on it.

Our modifications are concentrated in the `generate_inner` function in the `/src/vlm_interaction/VLMEvalKit/vlmeval/api/gpt.py` file. We pass the OpenRouter API key specified in the global config and the model to be called into this function, and add the corresponding Authorization field in the request header to implement the call to OpenRouter API.

We welcome community members to implement a lightweight VLM calling interface that does not depend on VLMEvalKit, so as to more conveniently integrate different VLM models.