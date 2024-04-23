
# 🤖 RAG Application 
< The only FREE TO USE RAG Application >
This is a RAG application made on the following technologies
- Gemini API
- Docker
- Python
- Streamlit
- Langchain
- A PDF :>

## 🪜Usage
Any amount of PDF can be stored inside a list and then their vector data will automatically be feeded to the Google Gemini Free Version using the 'embed-model-001' by Google. 

## 💪Requirement
- Google API Key need to be provided n the .env or You can directly upload in the Streamlit
- Also add "main.pdf" in the root directory, or you can change the name as well of this file.


## 📂Project Structure

- `app.py`: Main application script.
- `.env`: file which will contain your environment variable.
- `requirements.txt`: Python packages required for working of the app.
- `README.md`: Project documentation.

## 🐍Python Dependencies
- Streamlit
- google.generativeai
- dotenv
- langchain
- PyPDF2

## 🤖Running Guide
- Either Upload on Streamlit to run it
- or in the root directory open CMD and type "streamlit app.py"

## 📒Developer Notes
- This is right now hosted on Streamlit, but soon a Flask variant will be launched as well.
- Vercel / Netlify cannot support right now because of Langchain-Community module
- For any personal queries you can contact at mostuselessboy@gmail.com or https://www.linkedin.com/in/devhamzarizvi/
