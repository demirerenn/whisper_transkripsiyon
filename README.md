# Whisper Tabanlı Transkripsiyon Aracı

Bu repo, OpenAI Whisper tabanlı **ses → metin** dönüşümü için iki kullanım yolu sunar:

1) **Komut satırı**: `transcribe.py` ile dosyayı ver, çıktıları al.  
2) **Uygulama betiği**: `app.py` (opsiyonel) – yerel bir uygulama/başlatma betiği (ihtiyaçlarına göre genişletebilirsin).

## Özellikler
- MP3/WAV/M4A/FLAC vb. ses dosyaları
- **Dil tespiti** (manuel dil seçimi de mümkün)
- **Whisper model boyutları**: `tiny, base, small, medium, large-v2, large-v3`
- Çıktılar: **TXT**, **SRT**, **JSON**
- GPU varsa otomatik CUDA kullanımı

## Gereksinimler
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (PATH’te olmalı)
- Python paketleri: `openai-whisper`, `torch` (GPU istiyorsan uygun CUDA sürümüyle kur)

## Kurulum
```bash
git clone https://github.com/demirerenn/whisper_transkripsiyon.git
cd whisper_transkripsiyon

# (Önerilen) sanal ortam
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# Bağımlılıklar
pip install -r requirements.txt
# GPU için örnek:
# pip install torch --index-url https://download.pytorch.org/whl/cu121

## Lisans
Bu proje [MIT Lisansı](LICENSE) altında sunulmaktadır.
