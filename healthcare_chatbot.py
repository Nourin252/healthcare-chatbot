
import os
os.system('pip install --no-cache-dir transformers==4.24.0')
os.system('pip install --no-cache-dir torch')

# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import dateparser  # For date parsing
from datetime import datetime

# Load the pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Global variables to store user-provided details
appointment_info = {"doctor": None, "date": None}

# Function to validate and parse dates
def parse_date(user_input):
    parsed_date = dateparser.parse(user_input)
    if parsed_date and parsed_date >= datetime.now():
        return parsed_date.strftime("%Y-%m-%d")
    return None

# Chatbot function
def healthcare_chatbot():
    print("MedAI Chatbot: Hello! I am your virtual assistant. How can I help you today?")
    print("Type 'exit' to end the conversation.\n")

    # Chat history to maintain context
    chat_history_ids = None
    while True:
        # Get user input
        user_input = input("You: ")

        # Exit condition
        if user_input.lower() in ['exit', 'quit']:
            print("MedAI Chatbot: Thank you for using our service. Stay healthy!")
            break

        # Appointment Booking Intent
        if re.search(r"appointment|book|schedule", user_input, re.IGNORECASE):
            if appointment_info["doctor"] and appointment_info["date"]:
                print(f"MedAI Chatbot: Your appointment with Dr. {appointment_info['doctor']} has been scheduled for {appointment_info['date']}.")
            elif "doctor" not in appointment_info or not appointment_info["doctor"]:
                print("MedAI Chatbot: Sure, I can help you with appointment bookings. Please provide the doctor's name.")
                appointment_info["doctor"] = input("You: ").strip()
                print(f"MedAI Chatbot: Got it. Dr. {appointment_info['doctor']}. What date would you like to book?")
                date_input = input("You: ")
                parsed_date = parse_date(date_input)
                if parsed_date:
                    appointment_info["date"] = parsed_date
                    print(f"MedAI Chatbot: Your appointment with Dr. {appointment_info['doctor']} has been scheduled for {appointment_info['date']}.")
                else:
                    print("MedAI Chatbot: I couldn't understand the date. Please provide a valid date (e.g., 'next Monday' or '2025-01-10').")
            else:
                print("MedAI Chatbot: Please provide the preferred date for the appointment.")
                date_input = input("You: ")
                parsed_date = parse_date(date_input)
                if parsed_date:
                    appointment_info["date"] = parsed_date
                    print(f"MedAI Chatbot: Your appointment with Dr. {appointment_info['doctor']} has been scheduled for {appointment_info['date']}.")
                else:
                    print("MedAI Chatbot: I couldn't understand the date. Please provide a valid date (e.g., 'next Monday' or '2025-01-10').")

        # Symptoms or Health Advice Intent
        elif re.search(r"symptoms|advice|feeling unwell|cold|fever", user_input, re.IGNORECASE):
            print("MedAI Chatbot: I'm here to provide general advice, but please consult a doctor for a diagnosis.")
            print("MedAI Chatbot: What are your symptoms?")
            symptoms = input("You: ")
            print(f"MedAI Chatbot: Based on your symptoms ({symptoms}), I recommend resting, staying hydrated, and consulting a healthcare provider. If it's severe, please seek immediate medical attention.")

        # General Service Information Intent
        elif re.search(r"services|offerings|information", user_input, re.IGNORECASE):
            print("MedAI Chatbot: MedAI offers the following services:")
            print("1. Appointment scheduling")
            print("2. Teleconsultations")
            print("3. Health education resources")
            print("How else can I assist you?")

        # Fallback to AI Model for Freeform Responses
        else:
            input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
            bot_output = model.generate(
                input_ids,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            print(f"MedAI Chatbot: {response}")

# Run the chatbot
healthcare_chatbot()