from llama_cpp import Llama

llm = Llama(
    model_path="./gemma-2b-Q4_K_M.gguf",
    # chat_format="gemma",
    n_ctx=3000,
    n_threads=10,
    n_gpu_layers=10
)
print(
    llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "What's the capital of Colombia?"},
            {"role": "assistant", "content": "Bogota"},
            {"role": "user", "content": "What's the capital of Spain?"},
        ],
        max_tokens=5,
        temperature=0.1,  # Low temperature for less randomness
            stop=["</s>"],  # Stop token to indicate the end of a completion
            #echo=False  # Echo the prompt along with the model's response
    )
)
# {'id': 'chatcmpl-6037698e-2f05-496d-b49f-f7d61dde32f3', 'object': 'chat.completion', 'created': 1708599453, 'model': './models/gemma-7b-it-Q4_K_M.gguf', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'The answer is Madrid'}, 'finish_reason': 'length'}], 'usage': {'prompt_tokens': 37, 'completion_tokens': 4, 'total_tokens': 41}}