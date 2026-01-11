from transformers import pipeline


classifier = pipeline("sentiment-analysis")
res = classifier(
    ["I've been waiting for a HuggingFace course my whole life.", "I hate this so much!",
     "Oh great, another meeting that could have been an email.",
     "Sure, I love waiting in line for two hours just to get bad coffee.",
     "I can't say I'm thrilled about the new policy."]
)

print(res)


from transformers import TRANSFORMERS_CACHE
print(TRANSFORMERS_CACHE)
