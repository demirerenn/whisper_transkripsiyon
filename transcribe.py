# transcribe.py
import argparse
import os
import math
import torch
import whisper
from datetime import timedelta
from pathlib import Path
import json

def to_srt_timestamp(seconds: float) -> str:
    seconds = max(0, seconds)
    td = timedelta(seconds=seconds)
    # SRT format: HH:MM:SS,mmm
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int(td.microseconds / 1000)
    hours += td.days * 24
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def write_srt(segments, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = to_srt_timestamp(seg["start"])
            end = to_srt_timestamp(seg["end"])
            text = seg["text"].strip()
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

def main():
    parser = argparse.ArgumentParser(description="Local Whisper transcription")
    parser.add_argument("audio", help="Ses dosyası yolu (mp3/wav/m4a/flac/…)")
    parser.add_argument("--model", default="small", choices=[
        "tiny","base","small","medium","large-v2","large-v3"
    ], help="Whisper model boyutu (büyüdükçe doğruluk ↑, hız ↓)")
    parser.add_argument("--language", default=None,
                        help="Dil kodu (ör. 'tr','en'). Boş bırakılırsa otomatik tespit.")
    parser.add_argument("--output-dir", default="outputs", help="Çıktı klasörü")
    parser.add_argument("--temperature", type=float, default=0.0, help="Arama sıcaklığı (0–1)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[i] Cihaz: {device}")
    print(f"[i] Model yükleniyor: {args.model}")
    model = whisper.load_model(args.model, device=device)

    # fp16 yalnızca CUDA’da anlamlı
    fp16 = device == "cuda"

    print("[i] Çeviri başlıyor…")
    result = model.transcribe(
        str(audio_path),
        language=args.language,     # None ise otomatik dil tespiti
        task="transcribe",          # 'translate' = İngilizceye çeviri
        verbose=False,
        temperature=args.temperature,
        fp16=fp16
    )

    text = result.get("text", "").strip()
    segments = result.get("segments", [])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    txt_file = out_dir / f"{stem}.txt"
    srt_file = out_dir / f"{stem}.srt"
    json_file = out_dir / f"{stem}.json"

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    write_srt(segments, srt_file)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n=== TRANSKRİPT ===\n{text}\n")
    print(f"[✓] Kaydedildi:\n - {txt_file}\n - {srt_file}\n - {json_file}")

if __name__ == "__main__":
    main()
