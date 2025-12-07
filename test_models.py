from groq import Groq

client = Groq(api_key="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")

models = client.models.list()

for m in models.data:
    print(m.id)
