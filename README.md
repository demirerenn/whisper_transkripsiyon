# Whisper Tabanlı Yerel Transkripsiyon Aracı

Bu depo, OpenAI Whisper modelini kullanarak **ses dosyalarını metne** dönüştüren küçük bir komut satırı aracını içerir. Komut, tek bir `.py` dosyasıyla çalışır ve çıktı olarak **TXT**, **SRT** (altyazı) ve **JSON** verir.

## Özellikler
- MP3/WAV/M4A/FLAC vb. ses dosyalarını çözümler
- İsteğe bağlı **dil seçimi** (boş bırakırsanız otomatik tespit)
- Farklı **Whisper model boyutları** (tiny, base, small, medium, large-v2, large-v3)
- **TXT**, **SRT** ve **JSON** çıktıları
- GPU varsa otomatik CUDA kullanımı

## Gereksinimler
- Python 3.9+
- [FFmpeg](https://ffmpeg.org/) (Whisper için zorunlu – sistemine kurulu olmalı)
- Python paketleri: `openai-whisper`, `torch`

> Not: `torch` kurulumu işletim sistemine ve GPU desteğine göre değişir. Aşağıda örnek komutlar verilmiştir.

## Kurulum

1) Depoyu klonlayın veya indirin:
```bash
git clone https://github.com/<kullanici-adi>/<repo-adi>.git
cd <repo-adi>
```

2) Sanal ortam (önerilir):
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

3) Bağımlılıkları yükleyin:
```bash
# FFmpeg'i sisteminize kurun (OS'inize göre değişir)
# macOS (brew):   brew install ffmpeg
# Ubuntu/Debian:  sudo apt-get update && sudo apt-get install -y ffmpeg
# Windows:        https://www.gyan.dev/ffmpeg/builds/ (PATH'e ekleyin)

# Python paketleri
pip install -r requirements.txt
```

> GPU kullanacaksanız, PyTorch'u donanımınıza uygun şekilde kurun:
> - **CUDA** (örnek, CUDA 12.1): `pip install torch --index-url https://download.pytorch.org/whl/cu121`
> - **Sadece CPU**: `pip install torch`

## Kullanım

```bash
python transcribe.py <ses-dosyasi>   --model small   --language tr   --output-dir outputs   --temperature 0.0
```

### Parametreler
- `audio` (**zorunlu**): Ses dosyasının yolu. (mp3/wav/m4a/flac/…)
- `--model`: `tiny | base | small | medium | large-v2 | large-v3` (varsayılan: `small`)
- `--language`: ISO dil kodu. Örn. `tr`, `en`. Boş bırakırsanız otomatik tespit.
- `--output-dir`: Çıktı klasörü (varsayılan: `outputs`).
- `--temperature`: Arama sıcaklığı (0–1, varsayılan: `0.0`).

### Örnekler
```bash
# 1) Otomatik dil tespit ile transkripsiyon
python transcribe.py data/konusma.mp3

# 2) Türkçe dilini sabitleyerek küçük model ile
python transcribe.py data/soylesi.wav --language tr --model small

# 3) Daha yüksek doğruluk için medium / large-v3
python transcribe.py data/egitim.m4a --model large-v3
```

### Çıktılar
`outputs/` dizininde aynı dosya adıyla:
- `<isim>.txt` – düz metin transkript
- `<isim>.srt` – altyazı dosyası
- `<isim>.json` – Whisper ham sonuçları

## Proje Yapısı
```
.
├── transcribe.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Sorun Giderme
- **`ffmpeg not found`**: FFmpeg sistem PATH'inizde değil; kurup PATH'e eklediğinizden emin olun.
- **`CUDA` görülmüyor**: Uygun sürücü/CUDA kurulumu ve doğru PyTorch dağıtımını kullandığınızdan emin olun.
- **Yavaşlık**: Daha küçük modelleri (`tiny/base/small`) tercih edin veya GPU kullanın.
- **Türkçe noktalama/ayırma**: Model boyutunu artırmak çoğu zaman yardımcı olur.

## Lisans
Bu projeyi dilediğiniz gibi özelleştirebilirsiniz. Bir lisans eklemek isterseniz (örn. MIT), kök dizine `LICENSE` dosyası ekleyin.
