import os
import sys
import librosa
import numpy as np
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
from visualisation import generate_bokeh_plot

sys.path.insert(0, "/home/inarighas/Projects/voice-transcript/")
from app.main import process_transcription
from libs.main import compute_audio_features


# DESIGN implement changes to the standard streamlit UI/UX
# st.set_page_config(page_title="streamlit_audio_recorder")
# # Design move app further up and remove top padding
# st.markdown('''<style>.css-1egvi7u {margin-top: -3rem;}</style>''',
#             unsafe_allow_html=True
#             )
# # Design change st.Audio to fixed height of 45 pixels
# st.markdown('''<style>.stAudio {height: 45px;}</style>''',
#             unsafe_allow_html=True
#             )
# # Design change hyperlink href link color
# st.markdown('''<style>.css-v37k9u a {color: #ff4c4b;}</style>''',
#             unsafe_allow_html=True
#             )  # darkmode
# st.markdown('''<style>.css-nlntq9 a {color: #ff4c4b;}</style>''',
#             unsafe_allow_html=True
#             )  # lightmode

parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "st_audiorec/frontend/build")
st_audiorec = components.declare_component(
    "st_audiorec", path=build_dir
)

st.title("Voice Monitoring App Demo")
# st.markdown("by Ali Saghiran")
st.write(
    "You can record 🗣️ yourself down here ↓ and get instantaneous results."
)
st.write("Feel free to explore the app and the variables.")
st.write(f"Project path: `{os.getcwd()}`")

val = st_audiorec()

# st.write('Audio data received in the Python backend')
# will appear below this message ...')

if isinstance(val, dict):  # retrieve audio data
    with st.spinner("Retrieving audio-recording..."):
        ind, val = zip(*val["arr"].items())
        ind = np.array(ind, dtype=int)  # convert to np array
        val = np.array(val)  # convert to np array
        sorted_ints = val[ind]
        st.write(f"{val}")
        stream = BytesIO(
            b"".join([int(v).to_bytes(1, "big") for v in sorted_ints])
        )
        wav_bytes = stream.read()

    st.write("✔️ Thanks for recording yourself.")
    st.write("The audio file is successfully loaded.")
    # wav_bytes contains audio data in format to be further processed
    st.write(
        f"`Audio length: \t {len(wav_bytes)/(4*48000):.2f} seconds`"
    )
    st.write("`Quantization: \t 32bits`")
    st.write("`Sampling rate: \t 48kHz`")
    orig_SR = 48_000

    with st.spinner("Converting bytes to numpy array..."):
        int_buff = np.frombuffer(wav_bytes, dtype=np.int8)
        speech_arr = librosa.util.buf_to_float(wav_bytes, n_bytes=4)

    st.write("✔️ Converted successfully bytes to numerical array.")

    st.write("## Transcribing")
    with st.spinner("Feeding to Speech2text model:"):
        text, dur = process_transcription(speech_arr, orig_SR)
 
    word_rate = len(text.split(" ")) * 60 / (dur)
    st.write("✔️ Audio is successfully transcribed.")
    # st.audio(speech_arr, format='audio/wav')
    st.write(f"💬: «{text}».")
    st.write(f"Audio duration: {dur:.2f} seconds")
    st.write(f"Word rate: {word_rate:.2f} word/min")

    st.write("## Some features")
    with st.spinner("Feeding to speech processing pipeline:"):
        response = compute_audio_features(speech_arr, orig_SR)
    st.write("✔️ Audio is successfully analysed.")
    st.balloons()
    output = response.dict()
    output["SpeechRate_wpm"] = word_rate
    st.json(output)
    # normal values
    nominal_means = {
        "MeanPauseDuration": 0.47,
        "PauseFrequency": 9.43,
        "PauseVoiceRatio": 0.08,
        "equivalentSoundLevel_dBp": -35.6,
        "loudness_sma3_amean": 0.26,
        "loudness_std": 0.11,
        # "loudness_sma3_percentiles": [.13, .22, .36],
        "VoicedSegmentsPerSec": 2.45,
        "MeanUnvoicedSegmentLength": 0.16,
        "SpeechRate_wpm": 105,
        # "MeanVoicedSegmentLengthSec": 0.23
    }
    nominal_std = {
        "MeanPauseDuration": 0.432462,
        "PauseFrequency": 4.999438,
        "PauseVoiceRatio": 0.113542,
        "equivalentSoundLevel_dBp": 11.811862,
        "loudness_sma3_amean": 0.260724,
        "loudness_std": 0.70,
        # "loudness_sma3_percentiles": [0.136304, 0.232221, 0.374254],
        "VoicedSegmentsPerSec": 0.644294,
        "MeanUnvoicedSegmentLength": 0.267483,
        "SpeechRate_wpm": 30,
        # "MeanVoicedSegmentLengthSec": 0.096339
    }
    st.write(
        """The distributions in the background reflect the
                distributions in the general population.
                These were collected from spontaneous speech recordings
                of a significant set of French speakers.
                """
    )
    with st.spinner("Generating some plots:"):
        p = generate_bokeh_plot(output, nominal_means, nominal_std)
        for i in p.keys():
            st.write(f"### {i}")
            st.bokeh_chart(p[i], use_container_width=False)
