from transformers import pipeline

classifier = pipeline("sentiment-analysis")
print(classifier("Streamlit is now working fine!"))
