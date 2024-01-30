import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import random

# Update this to the path of your further fine-tuned model
model_name = './further_fine_tuned_model'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_travel_recommendation(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the first word from the response
    first_word = response.split()[2] if response else ""

    # Custom logic for specific outputs based on the first word
    if 'beaches' in first_word:
        return 'Goa'
    elif 'Historical' in first_word:
        return random.choice(['Taj Mahal Agra', 'Jaipur'])
    elif 'Adventure' in first_word:
        return 'Leh Ladakh'
    else:
        return first_word

st.title('Travel Destination Recommender')

# User input for preferences
user_preferences = st.text_input("Enter your travel preferences (e.g., 'beaches, historical sites'):")

if st.button('Recommend'):
    if user_preferences:
        # Generate the recommendation
        prompt = f"User preferences: {user_preferences}"
        recommendation = generate_travel_recommendation(prompt)
        st.write(f"Recommended Destination: {recommendation}")
    else:
        st.write("Please enter your travel preferences.")
