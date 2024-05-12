# sentiment_analysis_app.py
import json
import os
import glob
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import numpy as np
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Use the loaded environment variable
API_TOKEN = os.getenv("API_TOKEN")
# Hugging Face Inference API URL and API Token
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()
    except (KeyError, ValueError):
        return [{"generated_text": "An error occurred during text generation."}]

def generate_behavioral_analysis(messages,participant_name):
    # Combine messages into a single prompt
    prompt = (f"[INSTRUCTIONS] :Based ont the following messages of {participant_name}, please write a paragraph of 200 words about how they generally behave, their good points, bad points, and ways to improve: \n\n [MESSAGES] :" 
              + "\n\n".join(messages) + "\n\n[ANSWER] :")

    # Send the prompt to the Hugging Face model
    output = query({
        "inputs":prompt,
        }
    
    )
    #self.assertEqual(output, "Elie is a highly impulsive and restless individual who frequently sends multiple messages in quick succession, often containing attachments or requests for clarification on various topics. He is prone to getting easily distracted and jumping from one task or conversation to another, without fully completing what he has started. Elie's behavior can come off as erratic and frustrating to others, particularly when he fails to follow through or provide clear and concise answers.")
    return extract_answer(output)[0]

# Load all JSON files from a directory structure
def load_all_json_files(base_path):
    all_messages = []
    participant_set = set()

    inbox_path = os.path.join(base_path, 'your_instagram_activity', 'messages', 'inbox')
    json_files = glob.glob(os.path.join(inbox_path, '*', '*.json'))

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            all_messages.extend(data.get('messages', []))
            participants = data.get('participants', [])
            for participant in participants:
                participant_set.add(participant['name'])

    return list(participant_set), all_messages

# Sentiment Analysis using Hugging Face Pipeline
def sentiment_analysis(messages):
    sentiment_pipeline = pipeline('sentiment-analysis', 'cardiffnlp/twitter-roberta-base-sentiment-latest')
    if len(messages) > 0:
        return sentiment_pipeline(messages)
    else:
        return []

# Generate Statistics
def generate_statistics(messages):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(messages)
    word_counts = np.asarray(X.sum(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()
    word_count_dict = dict(zip(vocab, word_counts))
    sorted_words = sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_words

def extract_answer(data):
    answers = []
    for entry in data:
        # Extract the generated_text field from the entry
        text = entry.get("generated_text", "")
        # Find the starting index of the answer
        start_index = text.find("[ANSWER]")
        if start_index != -1:
            # Add the text starting from "[ANSWER]" to the list, trimming "[ANSWER] :"
            answers.append(text[start_index + len("[ANSWER] :"):].strip())
        else:
            # Append a None or some default text if "[ANSWER]" is not found
            answers.append("No answer found")
    return answers
# Streamlit Interactive UI
st.set_page_config(page_title="Messaging Analysis Tool", layout='wide')

st.title("Behavior Analysis and Sentiment Insights")
st.write("This tool analyzes the behavior and sentiment of participants in your Instagram messages.")
folder_path = st.text_input("Enter the base folder path containing your Instagram messages:")
if folder_path:
    participants, all_messages = load_all_json_files(folder_path)
    st.write(f"Loaded {len(all_messages)} messages from participants: {', '.join(participants)}")
    #store the messages from the others
    others_messages = [msg for msg in all_messages if msg['sender_name'] != 'Lancelot']
    #reduce to a single person
    all_messages = [msg for msg in all_messages if msg['sender_name'] == 'Lancelot']

    # Sentiment Analysis
    message_contents = [msg['content'] for msg in all_messages if 'content' in msg]
    #for messages of a single person
    sentiments = sentiment_analysis(message_contents)
    for i, sentiment in enumerate(sentiments):
        all_messages[i]['sentiment'] = sentiment

    st.subheader("Sentiment Analysis Results")
    average_sentiment_score_pos = np.mean([msg['sentiment']['score'] for msg in all_messages if 'sentiment' in msg and msg['sentiment']['label'] == 'positive'])
    print(average_sentiment_score_pos)
    print([msg['sentiment']['score'] for msg in all_messages if 'sentiment' in msg ])
    average_sentiment_score_neg = np.mean([msg['sentiment']['score'] for msg in all_messages if 'sentiment' in msg and msg['sentiment']['label'] == 'negative'])
    st.write(f"**Your Average Sentiment Score:** {((average_sentiment_score_pos-average_sentiment_score_neg)/2):.2f}")
    st.write(f"**Lancelot's gpt analysis:** {generate_behavioral_analysis(message_contents,'Lancelot')}")
    for participant in participants:
        omsg = [msg for msg in others_messages if msg['sender_name'] == participant]
        content= [msg['content'] for msg in omsg if 'content' in msg]
        sentiments = sentiment_analysis(content)
        for i, sentiment in enumerate(sentiments):
            omsg[i]['sentiment'] = sentiment
        sentiment_score_participant_neg = np.mean([msg['sentiment']['score'] for msg in omsg if 'sentiment' in msg and msg['sentiment']['label'] == 'negative'])
        sentiment_score_participant_pos = np.mean([msg['sentiment']['score'] for msg in omsg if 'sentiment' in msg and msg['sentiment']['label'] == 'positive'])
        st.write("---------------------------------------------------------------------------------------")
        st.write(f"**{participant}'s Average Sentiment Score:** {((sentiment_score_participant_pos-sentiment_score_participant_neg)/2):.2f}")
        st.write(f"**{participant}'s message received:** {len(content)}")
        if(len(content)>100):
            st.write(f"**{participant}'s gpt analysis:** {generate_behavioral_analysis(content,participant)}")


    # Statistics
    st.subheader("Message Behavior Statistics")
    stats = generate_statistics(message_contents)
    st.write("**Most Frequent Words:**")
    count=0
    unintersting_words = ['et', 'que', 'pas', 'je', 'mais', 'ca', 'est', 'les', 'la', 'le','en','tu','pour','mdrr','plus','si','des','il','ai','une','du','des','de','qui','bien','au','on','ce','avec','a','dans','ou','un','sur','ne','etre','faire','comme','c','t','y','moi','suis','quand','ma','mon','trop','elle','tout','fait','peut','aussi','vraiment','alors','sans','peux','ils','sont','vrai','es']
    for word, counts in stats:
         if count < 10:
            if word not in unintersting_words:
                st.write(f"- {word}: {counts}")
                count+=1
         else:
            break


            '''
    # Retrieval-Augmented Generation (RAG)
    st.subheader("Retrieval-Augmented Generation (RAG) Q&A")
    rag = RAG()
    query = st.text_input("Ask something about the messages:")
    if query:
        answer = rag.generate_response(query, message_contents)
        st.write(f"**Answer:** {answer}")
        '''