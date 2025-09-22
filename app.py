import streamlit as st
import whisper
import torch
from pathlib import Path
import tempfile
import time
from datetime import timedelta
import json


# Orijinal script'teki SRT formatlama fonksiyonu
def to_srt_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(td.microseconds / 1000)
    hours += td.days * 24
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


# Orijinal script'teki SRT yazma fonksiyonu
def generate_srt_content(segments):
    content = []
    for i, seg in enumerate(segments, start=1):
        start = to_srt_timestamp(seg["start"])
        end = to_srt_timestamp(seg["end"])
        text = seg["text"].strip()
        content.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(content)


# --- Arayüz Başlığı ve Açıklaması ---
st.set_page_config(page_title="Whisper Ses Deşifre Aracı", layout="wide")
st.title("🎙️ Whisper ile Sesi Metne Dönüştürme Aracı")
st.markdown("""
Bu araç, yüklediğiniz bir ses dosyasını (MP3, WAV, M4A vb.) OpenAI'nin Whisper modelini kullanarak metne dönüştürür.
Sonuç olarak deşifre edilmiş metni, SRT altyazı dosyasını ve detaylı JSON çıktısını indirebilirsiniz.
""")

# --- Çıktı Klasörü ---
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Kenar Çubuğu (Ayarlar) ---
st.sidebar.header("Ayarlar")

# 1. Dosya Yükleme Alanı
uploaded_file = st.sidebar.file_uploader(
    "Deşifre edilecek ses dosyasını seçin",
    type=['mp3', 'wav', 'm4a', 'flac']
)

# 2. Model Seçimi
model_size = st.sidebar.selectbox(
    "Whisper Model Boyutu",
    options=["tiny", "base", "small", "medium"], #"large-v3"
    index=2,  # Varsayılan olarak 'small' seçili gelsin
    help="Daha büyük modeller daha yavaştır ancak daha yüksek doğruluk sunar."
)

# 3. Dil Seçimi
language = st.sidebar.text_input(
    "Dil (isteğe bağlı)",
    placeholder="tr, en, de...",
    help="Boş bırakırsanız dil otomatik olarak tespit edilir. Örnek: 'tr' (Türkçe), 'en' (İngilizce)"
)
# Boş string yerine None gönderilmesi için kontrol
language = language if language else None

# --- Ana Ekran ---
if uploaded_file is not None:
    st.audio(uploaded_file, format=uploaded_file.type)

    if st.button("Deşifre Et", type="primary"):
        # Cihaz belirleme
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Kullanılan cihaz: **{device.upper()}**")

        try:
            # Modeli yükleme (önbelleğe alınır)
            with st.spinner(f"'{model_size}' modeli yükleniyor... (ilk seferde uzun sürebilir)"):
                model = whisper.load_model(model_size, device=device)
            st.success(f"'{model_size}' modeli başarıyla yüklendi.")

            # Dosyayı geçici bir konuma kaydet
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                audio_path = tmp.name

            # Deşifre işlemi
            transcription_start_time = time.time()
            with st.spinner("Deşifre işlemi sürüyor, lütfen bekleyin..."):
                result = model.transcribe(
                    audio_path,
                    language=language,
                    fp16=torch.cuda.is_available()
                )
            transcription_end_time = time.time()

            total_time = transcription_end_time - transcription_start_time
            st.success(f"Deşifre tamamlandı! Süre: {total_time:.2f} saniye")

            # --- Çıktı Gösterim Alanı ---
            st.header("Çıktılar")

            # 1. Deşifre Metni
            full_text = result.get("text", "").strip()
            st.subheader("Deşifre Metni")
            st.text_area("Metin", full_text, height=250)

            # Dosya isimlerini oluştur
            stem = Path(uploaded_file.name).stem

            # İndirme butonları için içerikleri hazırla
            srt_content = generate_srt_content(result.get("segments", []))
            json_content = json.dumps(result, ensure_ascii=False, indent=2)

            # 2. İndirme Linkleri
            st.subheader("Dosyaları İndir")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📝 Metin Dosyası (.txt)",
                    data=full_text.encode("utf-8"),
                    file_name=f"{stem}.txt",
                    mime="text/plain"
                )
            with col2:
                st.download_button(
                    label="🎬 Altyazı Dosyası (.srt)",
                    data=srt_content.encode("utf-8"),
                    file_name=f"{stem}.srt",
                    mime="text/plain"
                )
            with col3:
                st.download_button(
                    label="⚙️ JSON Dosyası (.json)",
                    data=json_content.encode("utf-8"),
                    file_name=f"{stem}.json",
                    mime="application/json"
                )

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

else:
    st.info("Lütfen sol taraftaki menüden bir ses dosyası yükleyin ve ayarları yapılandırın.")