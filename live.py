import cv2
import re
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import ollama
import keyboard
import os
import wave
import pyaudio
import threading

# Initialize the Ollama Mistral client
client = ollama.Client(host='http://localhost:11434')

# Global flag to control recording
recording = True

def record_audio(audio_path):
    global recording

    chunk = 1024  # Record in chunks of 1024 samples
    sample_format = pyaudio.paInt16  # 16 bits per sample
    channels = 1
    fs = 44100  # Record at 44100 samples per second

    p = pyaudio.PyAudio()  # Create an interface to PortAudio

    print("Recording audio.")
    stream = p.open(format=sample_format,
                    channels=channels,
                    rate=fs,
                    frames_per_buffer=chunk,
                    input=True)
    
    frames = []  # Initialize array to store frames

    while recording:
        data = stream.read(chunk)
        frames.append(data)
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    print("Audio recording stopped.")

    # Save the recorded data as a WAV file
    wf = wave.open(audio_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def record_video(video_path, audio_path):
    global recording

    cap = cv2.VideoCapture(0)  # Use 0 for the default camera

    # Define the codec and create VideoWriter object for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

    audio_thread = threading.Thread(target=record_audio, args=(audio_path,))
    audio_thread.start()

    print("Recording video. Press 'q' to stop recording.")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            out.write(frame)
            cv2.imshow('Recording', frame)
            if keyboard.is_pressed('q'):
                recording = False
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video recording stopped.")
    audio_thread.join()

def transcribe_audio_to_text(audio_path):
    recognizer = sr.Recognizer()
    
    # Check if the audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file {audio_path} not found.")
    
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
    
    prompt = f"You are an expert technical interviewer. Question the candidate based on his profile and his experience in the relevant field. Analyze the following candidate profile \n\n{text} and his skill level is {complexity}. Ask a technical question based on his experience:"
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
    video_path = "recorded_video.mp4"  # Path to save the recorded video
    audio_path = "extracted_audio.wav"   # Path to save the extracted audio
    
    # Record live video
    record_video(video_path, audio_path)
    
    # Transcribe audio to text
    text_content = transcribe_audio_to_text(audio_path)
    
    if not text_content:
        print("Audio transcription failed. Exiting.")
        return
    
    # Save transcription to a text file
    with open("transcription.txt", "w") as f:
        f.write(text_content)
    
    # Accumulate scores
    total_score = 0
    question_count = 0
    
    # Proceed with interview process using text_content
    experience_level = input("Are you an experienced professional or a fresher? (experienced/fresher): ").strip().lower()
    difficulty = int(input("Rate the difficulty level for the topic (1-5 stars): ").strip())

    follow_up_complexity = "simple" if experience_level == "fresher" else "detailed"

    # Ask three questions by default
    for _ in range(3):
        # Generate and ask a technical question
        question_response = generate_question(client, text_content, difficulty)
        question = question_response.get('response', "").strip()
        print(f"Question: {question}")
        
        answer = input("Your answer: ")
        score = validate_answer_llm(client, question, answer, text_content)
        total_score += score
        question_count += 1
        print(f"Score of your answer: {score}")
        
        # Ask for follow-up questions
        if _ < 2:  # Skip follow-up for the last question
            continue_session = input("Would you like to continue with a follow-up question? (yes/no): ").strip().lower()
            if continue_session != 'yes':
                break

            follow_up_prompt = f"Based on the answer '{answer}', ask a more {follow_up_complexity} technical question."
            follow_up_response = client.generate(model='mistral', prompt=follow_up_prompt)
            question = follow_up_response.get('response', "").strip()
            print(f"Follow-up Question: {question}")
            
            answer = input("Your answer: ")
            score = validate_answer_llm(client, question, answer, text_content)
            total_score += score
            question_count += 1
            print(f"Score of your answer: {score}")

    # Print overall score
    if question_count > 0:
        overall_score = total_score / question_count
        print(f"\nOverall Score: {overall_score:.2f}")
    else:
        print("\nNo questions were answered.")

if __name__ == "__main__":
    main()
