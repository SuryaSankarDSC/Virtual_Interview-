import re
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os
import ollama

# Initialize the Ollama Mistral client
client = ollama.Client(host='http://localhost:11434')

def extract_audio_from_video(video_path, audio_path):
    # Load video file
    video = VideoFileClip(video_path)
    
    # Extract audio and save as .wav
    video.audio.write_audiofile(audio_path, codec='pcm_s16le')  # 'pcm_s16le' codec writes raw .wav

def transcribe_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    # Load audio file
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
        
    # Recognize speech using Google Web Speech API
    try:
        text = recognizer.recognize_google(audio)
        print("Text Transcription:")
        print(text)
        return text
    except sr.UnknownValueError:
        print("Google Web Speech API could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Web Speech API; {e}")

def generate_question(client, text, difficulty):
    complexity_levels = ["very simple", "simple", "moderate", "complex", "very complex"]
    complexity = complexity_levels[difficulty - 1] if 1 <= difficulty <= 5 else "moderate"
    
    prompt = f"You are an expert technical interviewer. Question the candidate based on his profile and his experience in the relevant field. Analyze the following candidate profile \n\n{text} and his skill level is {complexity}. Ask technical question based on his experience:"
    response = client.generate(model='mistral', prompt=prompt)
    return response

def validate_answer_llm(client, question, answer, text):
    prompt = f"Given the following context:\n\n{text}\n\nEvaluate the answer '{answer}' to the question '{question}'. Provide a score between 0 and 100 based on its accuracy, relevance, and completeness."
    response = client.generate(model='mistral', prompt=prompt)
    score_str = response.get('response', "").strip()
    
    # Extract score using regular expression
    score = re.search(r'\b\d+\b', score_str)
    
    if score:
        return int(score.group())  # Convert score to integer
    else:
        return 0  # Return 0 if score extraction fails

def main():
    video_path = r"D:\Intern\0628.mp4"  # Path to your video file
    audio_path = "extracted_audiofinal.wav"   # Path to save the extracted audio
    
    # Extract audio from video
    extract_audio_from_video(video_path, audio_path)
    
    # Transcribe audio to text
    text_content = transcribe_audio_to_text(audio_path)
    
    if not text_content:
        print("Audio transcription failed. Exiting.")
        return
    
    # Save transcription to a text file
    with open("transcriptionfinal.txt", "w") as f:
        f.write(text_content)
    
    # Accumulate scores
    total_score = 0
    
    # Proceed with interview process using text_content
    experience_level = input("Are you an experienced professional or a fresher? (experienced/fresher): ").strip().lower()
    difficulty = int(input("Rate the difficulty level for the topic (1-5 stars): ").strip())

    follow_up_complexity = "simple" if experience_level == "fresher" else "detailed"

    # Generate and ask the initial technical question
    question_response = generate_question(client, text_content, difficulty)
    initial_question = question_response.get('response', "").strip()
    print(f"Initial Question: {initial_question}")
    
    initial_answer = input("Your answer: ")
    initial_score = validate_answer_llm(client, initial_question, initial_answer, text_content)
    total_score += initial_score
    print(f"Score of your answer: {initial_score}")

    # Ask follow-up technical questions based on the initial answer
    for _ in range(3):  # Number of follow-up questions can be adjusted
        follow_up_prompt = f"Based on the answer '{initial_answer}', ask a more {follow_up_complexity} technical question."
        follow_up_response = client.generate(model='mistral', prompt=follow_up_prompt)
        follow_up_question = follow_up_response.get('response', "").strip()
        print(f"Follow-up Question: {follow_up_question}")
        
        follow_up_answer = input("Your answer: ")
        follow_up_score = validate_answer_llm(client, follow_up_question, follow_up_answer, text_content)
        total_score += follow_up_score
        print(f"Score of your answer: {follow_up_score}")
    
    # Print overall score
    print(f"\nOverall Score: {total_score/4}")

if __name__ == "__main__":
    main()
