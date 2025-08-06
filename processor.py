import os
import cv2
import torch
import pandas as pd
import matplotlib.pyplot as plt
from moviepy import VideoFileClip
import speech_recognition as sr
from deepface import DeepFace
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from ultralytics import YOLO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from wordcloud import WordCloud

# ----------- Emotion Model -----------
def load_emotion_model():
    model_name = "cardiffnlp/twitter-roberta-base-emotion"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# ----------- Audio & Transcription -----------
def extract_audio(video_path):
    os.makedirs("outputs", exist_ok=True)
    clip = VideoFileClip(video_path)
    audio_out = os.path.join("outputs", os.path.basename(video_path).split('.')[0] + ".wav")
    clip.audio.write_audiofile(audio_out)
    return audio_out

def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except:
        return "Transcription failed."

# ----------- Emotion Analysis -----------
def analyze_emotion(text):
    tokenizer, model = load_emotion_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    labels = ["anger", "joy", "optimism", "sadness"]
    predicted_class = torch.argmax(probs).item()
    return labels[predicted_class], probs[0][predicted_class].item(), probs.tolist()[0]

def detect_video_emotion(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    success, frame = cap.read()
    while success:
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions.append(result[0]['dominant_emotion'])
        except:
            pass
        for _ in range(30):
            cap.read()
        success, frame = cap.read()
    cap.release()
    return max(set(emotions), key=emotions.count) if emotions else "Unknown"

def combine_emotions(transcript_emotion, video_emotion):
    return transcript_emotion if transcript_emotion.lower() == video_emotion.lower() else f"Mixed ({transcript_emotion} + {video_emotion})"

# ----------- Text Summarization -----------
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text[:1024], max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

# ----------- Word Cloud -----------
def generate_wordcloud(text, video_name):
    os.makedirs("outputs/wordclouds", exist_ok=True)
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    wc_path = f"outputs/wordclouds/{video_name}_wordcloud.png"
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(wc_path)
    plt.close()
    return wc_path

def generate_combined_wordcloud(transcripts):
    combined_text = " ".join(transcripts)
    return generate_wordcloud(combined_text, "combined")

# ----------- YOLOv8 Object Detection with Labeled Snapshots -----------
def detect_objects_yolov8(video_path):
    model = YOLO("C:\\Users\\AWS\\Desktop\\MCA\\4th Trimester\\AdvancedDataAnalytics\\VideoAnalysis\\yolov8n.pt")  # lightweight for CPU
    cap = cv2.VideoCapture(video_path)
    os.makedirs("outputs/snapshots", exist_ok=True)
    detections = []
    snapshots = []

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1
        if frame_index % 30 != 0:  # skip frames for speed
            continue

        results = model(frame, verbose=False)
        if results and results[0].boxes:
            boxes = results[0].boxes
            conf_scores = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            labels = [model.names[i] for i in cls_ids]

            # Draw bounding boxes
            for box, label, conf in zip(boxes.xyxy.cpu().numpy(), labels, conf_scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                detections.append((label, conf))

            snapshot_path = f"outputs/snapshots/{os.path.basename(video_path)}_{frame_index}.jpg"
            cv2.imwrite(snapshot_path, frame)
            snapshots.append((snapshot_path, sum(conf_scores)))

    cap.release()
    snapshots = sorted(snapshots, key=lambda x: x[1], reverse=True)[:5]
    object_counts = {}
    for label, _ in detections:
        object_counts[label] = object_counts.get(label, 0) + 1
    return object_counts, [s[0] for s in snapshots]

# ----------- Fusion: Text + Visual -----------
def classify_video_type(video_path, transcript):
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    labels = ["Accident", "Sports", "News", "Entertainment", "Protest", "Documentary"]
    text_result = classifier(transcript[:512], candidate_labels=labels)
    text_label, text_confidence = text_result['labels'][0], text_result['scores'][0]

    objects, snapshots = detect_objects_yolov8(video_path)
    visual_label = "Accident" if ("car" in objects or "truck" in objects) and objects.get("person", 0) < 20 else "Protest" if objects.get("person", 0) > 20 else text_label
    final_label = text_label if text_label == visual_label else ("Accident" if "Accident" in [text_label, visual_label] else visual_label)
    confidence = round((text_confidence + 0.9) / 2, 2)
    return final_label, confidence, objects, snapshots

# ----------- Q&A with Flan-T5 -----------
qa_model = pipeline("text2text-generation", model="google/flan-t5-large")

def ask_video_question(question, transcript, summary):
    prompt = f"Answer based on this video:\nSummary: {summary}\nTranscript: {transcript}\nQuestion: {question}\nAnswer in detail:"
    return qa_model(prompt, max_length=150)[0]['generated_text']

def ask_multi_video_question(question, summaries, transcripts):
    prompt = f"Here are multiple videos:\nSummaries: {' | '.join(summaries)}\nTranscripts: {' | '.join(transcripts)}\nQuestion: {question}\nAnswer based on all videos:"
    return qa_model(prompt, max_length=200)[0]['generated_text']

# ----------- Emotion Chart for PDF -----------
def generate_emotion_chart(distribution, video_name):
    labels = ["anger", "joy", "optimism", "sadness"]
    plt.figure()
    plt.bar(labels, distribution, color=['red', 'green', 'blue', 'purple'])
    plt.title(f"Emotion Distribution for {video_name}")
    chart_path = f"outputs/{video_name}_chart.png"
    plt.savefig(chart_path)
    plt.close()
    return chart_path

# ----------- PDF Report Generation -----------
def generate_pdf_report(video_data, pdf_path):
    chart_path = generate_emotion_chart(video_data['Distribution'], video_data['Video'])
    wordcloud_path = generate_wordcloud(video_data['Transcript'], video_data['Video'])

    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph(f"Video Analysis Report: {video_data['Video']}", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Video Type: {video_data['Video Type']} (Confidence: {video_data['Confidence']})", styles['Normal']))
    elements.append(Paragraph(f"Transcript Emotion: {video_data['Transcript Emotion']}", styles['Normal']))
    elements.append(Paragraph(f"Video Emotion: {video_data['Video Emotion']}", styles['Normal']))
    elements.append(Paragraph(f"Final Emotion: {video_data['Final Emotion']}", styles['Normal']))
    elements.append(Paragraph(f"Summary: {video_data['Summary']}", styles['Normal']))
    elements.append(Paragraph(f"Detected Objects: {video_data['Detected Objects']}", styles['Normal']))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Emotion Distribution:", styles['Heading2']))
    elements.append(Image(chart_path, width=400, height=225))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Word Cloud:", styles['Heading2']))
    elements.append(Image(wordcloud_path, width=400, height=225))

    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Snapshots:", styles['Heading2']))
    for snap in video_data['Snapshots']:
        elements.append(Image(snap, width=400, height=225))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return pdf_path

# ----------- Process Multiple Videos -----------
def process_videos(video_paths):
    results = []
    for video in video_paths:
        audio = extract_audio(video)
        transcript = transcribe_audio(audio)
        emotion_label, _, emotion_distribution = analyze_emotion(transcript)
        video_emotion = detect_video_emotion(video)
        summary = summarize_text(transcript)
        final_emotion = combine_emotions(emotion_label, video_emotion)
        video_type, confidence, objects, snapshots = classify_video_type(video, transcript)
        wordcloud_path = generate_wordcloud(transcript, os.path.basename(video))
        results.append({
            "Video": os.path.basename(video),
            "Video Type": video_type,
            "Confidence": confidence,
            "Detected Objects": objects,
            "Snapshots": snapshots,
            "Transcript": transcript,
            "Summary": summary,
            "Transcript Emotion": emotion_label,
            "Video Emotion": video_emotion,
            "Final Emotion": final_emotion,
            "Distribution": emotion_distribution,
            "WordCloud": wordcloud_path
        })
    return pd.DataFrame(results)
