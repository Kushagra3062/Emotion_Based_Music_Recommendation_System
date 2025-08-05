import streamlit as st
from src.emotion_detection.predict import get_emotion
from src.music.get_songs import MusicRecommender
import subprocess
import json
import shlex

def search_youtube_video_id(query):
    cmd = f"yt-dlp 'ytsearch1:{query}' --print-json --skip-download"
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)
        first_line = result.stdout.strip().split("\n")[0]
        data = json.loads(first_line)
        return data.get("id"), data.get("title")
    except Exception:
        return None, None

def display_song(tracks,artists):
    try:
        tracks_list = tracks.tolist()
    except AttributeError:
        tracks_list = list(tracks)
    try:
        artists_list = artists.tolist()
    except AttributeError:
        artists_list = list(artists)

    for t, a in zip(tracks_list, artists_list):
        
        query = f"{t} {a}"
        video_id, yt_title = search_youtube_video_id(query)

        iframe = (
        f'<iframe width="100%" height="200" src="https://www.youtube.com/embed/{video_id}?rel=0" '
        'frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" '
        'allowfullscreen></iframe>'
        if video_id else "<p style='color:gray;'>Video not found.</p>"
        )
        
        st.markdown(
            f"""
            <div style="
                border:1px solid #e2e8f0;
                padding:16px;
                border-radius:12px;
                max-width:500px;
                box-shadow:0 8px 30px rgba(0,0,0,0.05);
                margin-bottom:12px;
            ">
              <h3 style="margin:4px 0;">{t}</h3>
              <p style="margin:2px 0;"><strong>Artist:</strong> {a}</p>
              {iframe}
            </div>
            """,
            unsafe_allow_html=True,
        )
def main():
    st.set_page_config(page_title="Emotion Music Recommender", layout="centered")
    st.markdown(
        """
        <div style="background: linear-gradient(90deg,#4f46e5,#06b6d4); padding:18px; border-radius:12px;">
          <h1 style="margin:0; color:white; font-family: 'Segoe UI', system-ui;">🎶 Emotion-Based Music Recommendations</h1>
          <p style="margin:4px 0 0; color:white; font-size:14px;">Detect emotion and get matching songs with previews.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # spacing
    recommender = MusicRecommender(r'assest/dataset/ClassicHit.csv')
    
    col1,col2 = st.columns([3,1])
    with col1:
         st.markdown("### Capture emotion and fetch tailored tracks.")
    with col2:
        st.write("")
        capture = st.button("🎥 Capture Emotion",use_container_width = True)


    if capture:
        placeholder = st.empty()

        with st.spinner("Detecting emotion..."):
            placeholder.markdown(
                """
                <div style="border: 2px solid white; padding:18px; border-radius:12px;margin:10px;align-item:center;text-align:center;">
                    <p style="margin:4px 0 0; color:white; font-size:18px;">
                        Detecting emotion by observing 40 frames at 0.5 second FPS
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            emotion_list = get_emotion()
        placeholder.empty()
        if not emotion_list:
             st.warning("no emotion detected")
             return
        st.write("")
        st.success("Capture complete.")

        all_songs = []
        for emotion in emotion_list:
            st.subheader(f"Songs for: {emotion}")
            try:
                tracks,artist = recommender.recommend(emotion)
            except Exception as ex:
                st.error(f"Failed to get for '{emotion}': {ex}")
                continue
            display_song(tracks,artist) 
            any_displayed = True

        if not any_displayed:
            st.info("No recommendations found for detected emotions.")
if __name__ == "__main__":
    main()