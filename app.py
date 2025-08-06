import streamlit as st
import plotly.express as px
from processor import (
    process_videos, ask_video_question, ask_multi_video_question,
    generate_pdf_report, generate_combined_wordcloud
)

st.set_page_config(page_title="Video Analysis with Word Cloud & PDF", layout="wide")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Dashboard", "Ask Questions"])
uploaded_files = st.sidebar.file_uploader("Upload Videos", type=["mp4"], accept_multiple_files=True)

if "results" not in st.session_state and uploaded_files:
    with st.spinner("Analyzing videos..."):
        video_paths = []
        for file in uploaded_files:
            path = f"C:\\Users\\AWS\\Downloads\\analysis.mp4"
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            video_paths.append(path)
        st.session_state.results = process_videos(video_paths)

if "results" in st.session_state:
    df = st.session_state.results

    if page == "Dashboard":
        st.title("Video Analysis Dashboard")
        st.dataframe(df[["Video", "Video Type", "Confidence", "Transcript Emotion", "Video Emotion", "Final Emotion"]])

        st.subheader("Overall Emotion Distribution")
        labels = ["anger", "joy", "optimism", "sadness"]
        avg_distribution = [sum(d[i] for d in df["Distribution"]) / len(df) for i in range(4)]
        st.plotly_chart(px.bar(x=labels, y=avg_distribution, color=labels))

        st.subheader("Detected Objects, Snapshots, Word Cloud, and Reports")
        for i, video in enumerate(df["Video"]):
            st.markdown(f"### {video}")
            st.write(f"**Detected Objects:** {df['Detected Objects'][i]}")
            st.write(f"**Classification:** {df['Video Type'][i]} (Confidence: {df['Confidence'][i]})")
            st.image(df["WordCloud"][i], caption=f"Word Cloud for {video}", width=400)
            for snap in df["Snapshots"][i]:
                st.image(snap, width=300)

            pdf_path = f"outputs/{video}_report.pdf"
            if st.button(f"Generate PDF for {video}"):
                video_data = df.iloc[i].to_dict()
                generate_pdf_report(video_data, pdf_path)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(f"Download PDF for {video}", pdf_file, file_name=f"{video}_report.pdf")

        st.subheader("Download Analysis Data")
        st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "video_analysis.csv", "text/csv")
        st.download_button("Download JSON", df.to_json(orient='records'), "video_analysis.json", "application/json")

    elif page == "Ask Questions":
        st.title("Ask Questions")
        mode = st.radio("Choose mode:", ["Single Video", "Multi-Video"])
        if mode == "Single Video":
            selected_video = st.selectbox("Choose a video", df["Video"])
            question = st.text_input("Enter your question:")
            if st.button("Ask"):
                video_data = df[df["Video"] == selected_video].iloc[0]
                st.write(ask_video_question(question, video_data["Transcript"], video_data["Summary"]))
        else:
            question = st.text_input("Ask a question about all videos:")
            if st.button("Ask All"):
                st.write(ask_multi_video_question(question, df["Summary"].tolist(), df["Transcript"].tolist()))
                combined_wc = generate_combined_wordcloud(df["Transcript"].tolist())
                st.image(combined_wc, caption="Combined Word Cloud for All Videos", width=500)
else:
    st.write("Upload videos using the sidebar.")
