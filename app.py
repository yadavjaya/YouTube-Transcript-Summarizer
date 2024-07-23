from flask import Flask, request, jsonify, send_file
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
from language_tool_python import LanguageTool
import torch
from datetime import timedelta
import re
from transformers import BartForConditionalGeneration, BartTokenizer
from keybert import KeyBERT
import spacy
from collections import Counter
from googletrans import Translator as GoogleTranslator
#from docx import Document
from datetime import timedelta, datetime  # Import datetime module
import firebase_admin
from firebase_admin import initialize_app, db, firestore
from firebase_admin import credentials
from flask_cors import CORS
from io import BytesIO


app = Flask(__name__)



def error_response(message, status_code):
    return jsonify({"error": message}), status_code

def calculate_chunk_size(video_duration):
    if video_duration <= 600:
        return 800
    elif video_duration <= 1800:
        return 1200
    elif video_duration <= 3600:
        return 1500
    else:
        return 2000  

@app.route('/summary', methods=['GET'])
def summary_api():
    print("Request accepted")
    url = request.args.get('url', '')
    max_length = int(request.args.get('max_length', 150))  # Default max_length is 150
    video_id = extract_video_id(url)
    target_lang = request.args.get('target_lang', 'en')  # Default target language is English
    
    if not video_id:
        return error_response("Invalid video URL", 400)

    transcript = get_transcript(video_id)
    if not transcript:
        return error_response("Transcript not available", 404)

    video_info = YouTubeTranscriptApi.get_transcript(video_id)
    video_duration = get_video_duration(video_info)
    chunk_size = calculate_chunk_size(video_duration)
    summary_data = get_summary(transcript, chunk_size, max_length, target_lang)
    
    if not summary_data:
        return error_response("Error generating summary", 500)

    return jsonify(summary_data), 200

def extract_video_id(url):
    try:
        if 'youtube.com/watch' in url:
            video_id = url.split('v=')[1].split('&')[0]
        elif 'youtu.be' in url:
            video_id = url.split('/')[-1]
        else:
            video_id = None
        return video_id
    except Exception as e:
        print(f"Error extracting video ID: {str(e)}")
        return None

def get_video_duration(video_info):
    try:
        video_duration = 0
        for entry in video_info:
            if 'start' in entry and 'duration' in entry:
                start_time = parse_duration(entry['start'])
                duration = parse_duration(entry['duration'])
                end_time = start_time + duration
                
                if end_time.total_seconds() > video_duration:
                    video_duration = end_time.total_seconds()
        return video_duration
    except Exception as e:
        print(f"Error getting video duration: {str(e)}")
        return 0

def parse_duration(duration):
    if isinstance(duration, str):
        match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration)
        if match:
            hours = int(match.group(1)[1:]) if match.group(1) else 0
            minutes = int(match.group(2)[1:]) if match.group(2) else 0
            seconds = int(match.group(3)[1:]) if match.group(3) else 0
            return timedelta(hours=hours, minutes=minutes, seconds=seconds)
    elif isinstance(duration, float):
        return timedelta(seconds=duration)
    else:
        raise ValueError(f"Invalid duration format: {duration}")

def get_transcript(video_id):
    try:
        transcript_list = None
        preferred_languages = ['en', 'hi', 'mr', 'ta']  # English, Hindi, Marathi, Tamil
       

        
        # Try to get the transcript in preferred languages
        for lang in preferred_languages:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
            if transcript_list:
                break  # Break the loop if transcript is found
        
        # If no transcript is found in preferred languages
        if not transcript_list:
            print("Transcript not available in preferred languages")
            return None
        
        # Concatenate the text from all segments in the transcript
        transcript = ' '.join([d['text'] for d in transcript_list])
        return transcript
    
    except Exception as e:
        print(f"Error fetching transcript: {str(e)}")
        return None

@app.route('/download_summary', methods=['GET'])
def download_summary():
    url = request.args.get('url', '')
    endpoint = request.args.get('endpoint', 'summary')
    max_length = int(request.args.get('max_length', 150))  # Default max_length is 150
    target_lang = request.args.get('target_lang', 'en')  # Default target language is English

    video_id = extract_video_id(url)
    if not video_id:
        return error_response("Invalid video URL", 400)

    transcript = get_transcript(video_id)
    if not transcript:
        return error_response("Transcript not available", 404)

    if endpoint == 'summary':
        summary_data = get_summary(transcript, calculate_chunk_size(get_video_duration()), max_length, target_lang)
    else:
        summary_data = get_abstractive_summary(transcript, target_lang)

    if not summary_data:
        return error_response("Error generating summary", 500)

    summary = summary_data['summary']

    # Generate PDF from summary text
    pdf = generate_pdf(summary)

    # Create BytesIO object to store PDF content
    pdf_bytes = BytesIO()
    pdf.write(pdf_bytes)
    pdf_bytes.seek(0)

    filename = f"summary_{video_id}.pdf"
    return send_file(pdf_bytes, as_attachment=True, attachment_filename=filename, mimetype='application/pdf')

def generate_pdf(text):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import SimpleDocTemplate
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    # Create a canvas
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    paragraph = Paragraph(text, styles['Normal'])

    # Add the paragraph to the canvas
    paragraph.wrapOn(c, 400, 800)
    paragraph.drawOn(c, 100, 750)

    # Save the canvas as PDF
    c.save()

    return pdf_buffer
def get_summary(transcript, chunk_size, max_length, target_lang):
    try:
        summariser = pipeline("summarization", model="t5-small", truncation=True, revision="d769bba", return_tensors=False)
        summary_data = {'summary': '', 'keywords': []}

        for i in range(0, len(transcript), chunk_size):
            chunk = transcript[i:i + chunk_size]
            adjusted_max_length = min(max_length, len(chunk))
            min_length = min(adjusted_max_length, 30)

            summary_result = summariser(chunk, max_length=adjusted_max_length, min_length=min_length)[0]
            summary_text = summary_result['summary_text']

            summary_data['summary'] += summary_text

        keywords = extract_keywords(summary_data['summary'], target_lang)
        summary_data['keywords'] = keywords

        if target_lang != 'en':
            summary_data['summary'] = translate_summary(summary_data['summary'], target_lang)

        return summary_data
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        return None

def extract_keywords(text, target_lang, top_n=5):
    try:
        model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords = model.extract_keywords(text, top_n=top_n)

        # Translate the keywords to the target language
        translator = GoogleTranslator()
        translated_keywords = [translator.translate(keyword, dest=target_lang).text for keyword, _ in keywords]
        
        return translated_keywords
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return []

@app.route('/translate_keyword', methods=['GET'])
def translate_keyword_route():
    keyword = request.args.get('keyword')
    target_lang = request.args.get('target_lang')
    translated_keyword = translate_keyword(keyword, target_lang)
    return translated_keyword

def translate_keyword(keyword, target_lang):
    translator = GoogleTranslator()
    translated_keyword = translator.translate(keyword, dest=target_lang).text
    return translated_keyword

def translate_summary(summary, target_lang):
    translator = GoogleTranslator()
    chunk_size = 500
    translated_chunks = []
    for i in range(0, len(summary), chunk_size):
        chunk = summary[i:i+chunk_size]
        translated_chunk = translator.translate(chunk, dest=target_lang).text
        translated_chunks.append(translated_chunk)
    translated_summary = ''.join(translated_chunks)
    return translated_summary

def get_abstractive_summary(transcript, target_lang):
    try:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
        device = torch.device('cpu')

        inputs = tokenizer.encode("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True, padding=True).to(device)
        
        summary_ids = model.generate(inputs, max_length=700 , length_penalty=2.0, num_beams=4, early_stopping=True)
        abstractive_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        translated_summary = translate_summary(abstractive_summary, target_lang)

        keywords = extract_keywords(translated_summary, target_lang)
        
        summary_data = {'summary': translated_summary, 'keywords': keywords}
        return summary_data
    except Exception as e:
        print(f"Error generating abstractive summary: {str(e)}")
        return None

@app.route('/abstractive_summary', methods=['GET'])
def abstractive_summary_api():
    url = request.args.get('url', '')
    target_lang = request.args.get('target_lang', 'en')  # Default target language is English
    
    # Extract video ID from URL
    video_id = extract_video_id(url)
    if video_id:
        # Get transcript for the video
        transcript = get_transcript(video_id)
        if transcript:
            # Generate abstractive summary
            abstractive_summary_data = get_abstractive_summary(transcript, target_lang)
            
            if abstractive_summary_data:
                translated_summary = abstractive_summary_data['summary']
                keywords = extract_keywords(translated_summary, target_lang)
                
                # Add download URL to the summary data
                abstractive_summary_data['download_url'] = f"/download_summary?url={url}&target_lang={target_lang}&endpoint=abstractive_summary"

                # Include keywords in the returned JSON response
                abstractive_summary_data['keywords'] = keywords
        
                return jsonify(abstractive_summary_data), 200
            else:
                return error_response("Error generating abstractive summary", 500)
    else:
        return error_response("Invalid video URL", 400)
    

cred = credentials.Certificate('C:/Users/hp/Downloads/YTS/try/summary.json')

CORS(app)

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        
        print("Enter Submit feedback:")
        print("Request:", request.json)
        
        feedback_data = request.json
        if not feedback_data:
            return jsonify({"error": "No data provided"}), 400
        
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://summary-1e0f7-default-rtdb.firebaseio.com'})
        # Get a Realtime Database reference
        ref = db.reference('/')
        new_data_ref = ref.child('feedback').push(feedback_data)

        return jsonify({"message": "Feedback submitted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)