import ollama 

model = "qwen3:1.7b"

response = ollama.chat(
    model=model,
    messages=[
        {
            "role": "system",
            "content": (
                "You are a chemistry assistant specializing in UV-Vis spectroscopy" 

            )
        },
        {
            "role": "user",
            "content": (
                "Identify the main function group in ethanol, (SMILES string for reference - CCO). "
            )
        }
    ]
)

print(
    response["message"]["content"]
)