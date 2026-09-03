import streamlit as st
import whisper
import os


@st.cache_resource(ttl='1h')
def load_model(model:str) -> whisper.model.Model:
    return whisper.load_model(model)


@st.cache_resource(ttl='1h')
def transcript(model: str, file_path: str) -> str:
    model_instance = load_model(model)
    result = model_instance.transcribe(file_path)
    return result


st.set_page_config(page_title="App de Transcrição", page_icon="📝")
st.title("App de Transcrição")
st.markdown("Este aplicativo permite transcrever vídeos e aúdios em texto usando o modelo OpenAI Whisper.")

os.makedirs("./data",exist_ok=True)
files = os.listdir("./data")
files = [f for f in files if f.endswith((".mp4", ".mov", ".avi", ".mp3"))]

c1, c2, c3 = st.columns(3)

with c1:
    uploaded_file = st.selectbox("Escolha um arquivo", ["Selecionar arquivo..."] + files)
if uploaded_file != "Selecionar arquivo...":
    uploaded_file = os.path.join("./data", uploaded_file)
else:
    uploaded_file = None


with c2:
    models = ["tiny", "base", "small", "medium", "large"]
    model = st.selectbox("Escolha o modelo Whisper", models, index=len(models)-1)

with c3:
    time_mark = st.checkbox("Marcação de tempo")

if uploaded_file is not None:


    result = transcript(model, uploaded_file)

    if time_mark:
        txt = ""
        for segment in result["segments"]:
            txt += f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"
        st.text_area("Transcrição", txt, height=300)

    else:
        result = transcript(model, uploaded_file)
        txt = result["text"]
        st.text_area("Transcrição", txt, height=300)


    new_file = uploaded_file.split(".")[:-1]
    new_file = ".".join(new_file) + ".txt"
    if st.button("Baixar Transcrição"):
        with open(new_file, "w", encoding="utf-8") as f:
            f.write(txt if time_mark else result["text"])