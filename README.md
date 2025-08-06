# Video Sentiment & Classification Analysis
AI-powered video analysis combining YOLOv8 for object detection, speech-to-text, NLP-based sentiment and summarization, word clouds, emotion charts, Flan-T5 Q&amp;A, and PDF reports, all integrated into an interactive Streamlit dashboard.

This project performs **end-to-end video analysis** using **NLP + Computer Vision**, integrating:
- **Video classification** (Accident, Protest, Sports, etc.) using **YOLOv8** + **Zero-Shot NLP**.
- **Emotion detection** (from transcript & faces).
- **Text summarization**.
- **Word Clouds** based on transcripts.
- **Emotion Distribution Charts**.
- **PDF Reports** per video (classification, charts, snapshots, word cloud).
- **Interactive Q&A** (single & multi-video) using **Flan-T5**.
- **Streamlit Dashboard** for visualization, downloads, and chat.

---

## **Features**
1. **Upload multiple videos** via Streamlit.
2. Extracts:
   - Audio → Transcription (Google Speech Recognition)
   - Emotions (Text + Facial)
   - Key Objects (YOLOv8, CPU-friendly)
3. Generates:
   - **Summaries**
   - **Word Clouds**
   - **Emotion Distribution Charts**
   - **Snapshots of detected objects**
   - **PDF Report per video**
4. **Interactive Q&A**:
   - Single video questions.
   - Multi-video questions + combined word cloud.
5. Download **CSV**, **JSON**, and **PDFs** for reports.

---

## **Tech Stack**
- **Python 3.10+**
- **Streamlit** for dashboard.
- **Transformers (Hugging Face)** for NLP (Zero-Shot, Flan-T5, Summarization).
- **YOLOv8 (Ultralytics)** for object detection.
- **DeepFace** for emotion recognition.
- **MoviePy** + **SpeechRecognition** for audio extraction & transcription.
- **Matplotlib** + **WordCloud** for visualizations.
- **ReportLab** for PDF generation.

---

## **Setup**

### 1. Clone Repository
```bash
git clone https://github.com/your-username/video-sentiment-analysis.git
cd video-sentiment-analysis
