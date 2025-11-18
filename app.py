# REQUIREMENTS:
# streamlit==1.28.1
# requests==2.31.0
# python-dotenv==1.0.0
#
# Install with: pip install -r requirements.txt
# Run with: streamlit run app.py

import os
import streamlit as st
import requests
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Get API keys from Streamlit secrets or environment variables
try:
    ASSEMBLE_API_KEY = st.secrets.get("ASSEMBLE_API_KEY") or os.getenv("ASSEMBLE_API_KEY")
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
except:
    ASSEMBLE_API_KEY = os.getenv("ASSEMBLE_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

st.set_page_config(page_title="VARMASCRIBE", layout="wide")
st.markdown("<h1 style='text-align: center; color: #1e40af; font-size: 3em; margin-bottom: 0.5em;'>🎙️ VARMASCRIBE</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #64748b; font-size: 1.1em; margin-top: 0;'>AI-Powered Medical Transcription & Documentation</p>", unsafe_allow_html=True)

def upload_to_assemblyai(file_bytes, filename):
    upload_url = "https://api.assemblyai.com/v2/upload"
    headers = {"authorization": ASSEMBLE_API_KEY}

    response = requests.post(
        upload_url,
        headers=headers,
        data=file_bytes
    )
    if response.status_code != 200:
        st.error("Error uploading to AssemblyAI")
        st.write(response.text)
        return None

    return response.json()['upload_url']

def transcribe_with_assemblyai(audio_url: str):
    headers = {"authorization": ASSEMBLE_API_KEY}

    config = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "format_text": True,
        "punctuate": True,
        "speech_model": "universal",
        "language_detection": True
    }

    response = requests.post("https://api.assemblyai.com/v2/transcript", json=config, headers=headers)
    transcript_id = response.json()['id']
    polling_endpoint = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"

    with st.status("Transcribing audio with AssemblyAI…", expanded=True) as status:
        while True:
            result = requests.get(polling_endpoint, headers=headers).json()

            if result['status'] == 'completed':
                status.update(label="Transcription completed!", state="complete")
                return result

            elif result['status'] == 'error':
                status.update(label="Transcription failed", state="error")
                raise RuntimeError(result['error'])

            else:
                time.sleep(2)

def call_gemini(prompt: str):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    response = requests.post(url, json=payload)

    try:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        st.write("Gemini API Error:", response.text)
        return ""

def format_timestamp(milliseconds):
    """Convert milliseconds to MM:SS format"""
    if milliseconds is None:
        return "00:00"
    seconds = milliseconds // 1000
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def display_transcript_with_speakers(transcript_data):
    """Display transcript with speaker diarization in an enhanced format"""
    if isinstance(transcript_data, str):
        # Plain text transcript without speaker data
        st.markdown(transcript_data)
        return
    
    # Extract utterances with speaker labels
    utterances = transcript_data.get('utterances', [])
    
    if not utterances:
        # Fallback to plain text
        st.markdown(transcript_data.get('text', ''))
        return
    
    # Speaker colors for visual distinction
    speaker_colors = {
        0: "#3B82F6",  # Blue
        1: "#10B981",  # Green
        2: "#F59E0B",  # Amber
        3: "#EF4444",  # Red
        4: "#8B5CF6",  # Purple
        5: "#EC4899",  # Pink
    }
    
    # Create speaker labels
    speaker_labels = {}
    speaker_count = {}
    
    for utterance in utterances:
        speaker_id = utterance.get('speaker')
        if speaker_id is not None and speaker_id not in speaker_labels:
            speaker_num = len(speaker_labels)
            speaker_labels[speaker_id] = f"Speaker {chr(65 + speaker_num)}"  # A, B, C, etc.
            speaker_count[speaker_id] = 0
    
    # Display transcript with enhanced formatting
    st.markdown("<div style='background-color: #f8f9fa; border-radius: 12px; padding: 24px; border-left: 4px solid #1e40af;'>", unsafe_allow_html=True)
    
    for utterance in utterances:
        speaker_id = utterance.get('speaker')
        text = utterance.get('text', '')
        start_time = utterance.get('start')
        end_time = utterance.get('end')
        
        if speaker_id is not None:
            speaker_label = speaker_labels[speaker_id]
            color = speaker_colors.get(speaker_id, "#6B7280")
            timestamp = f"{format_timestamp(start_time)} - {format_timestamp(end_time)}"
            
            # Create speaker badge and text
            st.markdown(
                f"""<div style='margin-bottom: 16px;'>
                    <span style='display: inline-block; background-color: {color}; color: white; padding: 4px 12px; border-radius: 4px; font-weight: bold; font-size: 12px; margin-right: 8px;'>
                        {speaker_label}
                    </span>
                    <span style='color: #6B7280; font-size: 12px; margin-right: 12px;'>
                        {timestamp}
                    </span>
                    <br/>
                    <p style='margin: 8px 0 0 0; color: #1F2937; line-height: 1.6;'>{text}</p>
                </div>""",
                unsafe_allow_html=True
            )
    
    st.markdown("</div>", unsafe_allow_html=True)

windsurf_prompt_template = """
You are an expert medical scribe. Based on the following medical consultation transcript, generate comprehensive medical documentation in the following format:

## SOAP NOTE

### Subjective
- Chief Complaint
- History of Present Illness
- Past Medical History
- Medications
- Allergies
- Social History

### Objective
- Vital Signs
- Physical Examination
- Lab Results (if mentioned)

### Assessment
- Primary Diagnosis
- Differential Diagnoses
- Clinical Impression

### Plan
- Treatment Plan
- Medications
- Follow-up
- Patient Education

## HISTORY & PHYSICAL (H&P)

### History of Present Illness
### Past Medical History
### Medications
### Allergies
### Social History
### Review of Systems
### Physical Examination
### Assessment and Plan

---

Transcript to process:
{{TRANSCRIPT_HERE}}

Generate professional medical documentation based on this transcript. Use proper medical terminology and formatting.
"""

# Custom CSS for better styling
st.markdown("""
<style>
    .input-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .settings-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .feature-badge {
        display: inline-block;
        background-color: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("<div class='input-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>📁 Input Source</h3>", unsafe_allow_html=True)
    input_method = st.radio("Choose input method:", ["Upload Audio", "Paste Transcript"], horizontal=False, label_visibility="collapsed")
    
    if input_method == "Upload Audio":
        audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "m4a", "ogg", "flac"])
        transcript_text = ""
    else:
        audio = None
        transcript_text = st.text_area("Paste your transcript here:", height=250, placeholder="Enter medical consultation transcript...")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='settings-card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 0;'>⚙️ Documentation Options</h3>", unsafe_allow_html=True)
    include_soap = st.checkbox("📋 Generate SOAP Note", value=True)
    include_hp = st.checkbox("📝 Generate H&P Report", value=True)
    
    st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.3);'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-top: 1rem;'>🚀 Process</h3>", unsafe_allow_html=True)
    run = st.button("Generate Documentation", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

if run:
    final_transcript = transcript_text.strip() if transcript_text else ""
    transcript_data = None

    if audio and not final_transcript:
        st.info("📤 Uploading audio to AssemblyAI...")
        file_bytes = audio.read()
        audio_url = upload_to_assemblyai(file_bytes, audio.name)

        if audio_url:
            transcript_data = transcribe_with_assemblyai(audio_url)
            final_transcript = transcript_data.get('text', '') if isinstance(transcript_data, dict) else transcript_data
    elif transcript_text:
        final_transcript = transcript_text

    if not final_transcript:
        st.error("❌ No transcript found. Please upload audio or paste a transcript.")
        st.stop()

    # Display transcript with speaker diarization
    st.markdown("---")
    st.markdown("<h2 style='color: #1e40af;'>🎤 Transcript with Speaker Diarization</h2>", unsafe_allow_html=True)
    if transcript_data and isinstance(transcript_data, dict):
        display_transcript_with_speakers(transcript_data)
    else:
        st.markdown(final_transcript)

    full_prompt = windsurf_prompt_template.replace("{{TRANSCRIPT_HERE}}", final_transcript)

    with st.spinner("🤖 Generating medical documentation..."):
        result = call_gemini(full_prompt)

    if result:
        st.success("✅ Documentation generated successfully!")
        st.markdown("---")
        st.markdown("<h2 style='color: #1e40af;'>📄 Generated Medical Documentation</h2>", unsafe_allow_html=True)
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["📋 Full Document", "📥 Download"])
        
        with tab1:
            st.markdown(result)
        
        with tab2:
            st.markdown("### Download Options")
            col_md, col_txt = st.columns(2)
            with col_md:
                st.download_button(
                    label="📥 Download as Markdown",
                    data=result,
                    file_name="medical_documentation.md",
                    mime="text/markdown",
                    use_container_width=True
                )
            with col_txt:
                st.download_button(
                    label="📄 Download as Text",
                    data=result,
                    file_name="medical_documentation.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        st.error("❌ Failed to generate documentation. Please try again.")