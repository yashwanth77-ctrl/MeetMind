
from services.qa import client

models = client.models.list()

print("\nAVAILABLE GROQ MODELS:\n")

for model in models.data:
    print(model.id)
