"""
جسر الاتصال بين Arduino والسيرفر - مشروع "نخلة"
================================================
يشتغل على الكمبيوتر المتصل بـ Arduino عبر USB:
1. يقرأ عينات الصوت المرسلة من nakhla_sensor_uno.ino عبر Serial
2. يبنيها كملف WAV
3. يرسله لسيرفر FastAPI (/predict-full) ويطبع النتيجة

الاستخدام:
    pip install pyserial requests
    python serial_bridge.py COM3        # على ويندوز
    python serial_bridge.py /dev/ttyUSB0  # على لينكس/ماك

اضغطي زر الفحص المتصل بـ Arduino (أو اكتبي R بالمنفذ التسلسلي)
لبدء تسجيل جديد.
"""
import os
import sys
import time
import numpy as np
import requests
import serial
from scipy.io import wavfile

SAMPLE_RATE = 4000  # يجب أن يطابق SAMPLE_RATE بملف الأردوينو وfeatures.py
BAUD = 250000       # يجب أن يطابق Serial.begin بملف الأردوينو
SERVER_URL = os.environ.get("NAKHLA_SERVER_URL", "http://localhost:8000").rstrip("/") + "/predict-full"
# مفتاح الجهاز (نفس المفتاح داخل الباركود): الموقع لا يقبل قراءات الجهاز بدونه. ضعه بمتغير البيئة NAKHLA_KEY
DEVICE_KEY = os.environ.get("NAKHLA_KEY", "")
DEFAULT_PALM_ID = "Palm-A12"  # يستخدم بس لو ضغطتِ Enter بدون كتابة رقم جديد


def read_one_recording(ser):
    """يقرأ تسجيل واحد كامل بين إشارتي START و END من Arduino."""
    samples = []
    print("في انتظار بدء التسجيل (يرسل الجسر الأمر تلقائيًا؛ وزر D2 اختياري)...")
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line == "START":
            print("جارٍ استقبال العينات...")
            samples = []
        elif line == "END":
            # الأردوينو يطبع بعدها ELAPSED_MS,<ملّي ثانية>: زمن التسجيل الفعلي
            elapsed = None
            nxt = ser.readline().decode(errors="ignore").strip()
            if nxt.startswith("ELAPSED_MS,"):
                try: elapsed = int(nxt.split(",")[1])
                except ValueError: pass
            print(f"اكتمل التسجيل: {len(samples)} عينة" + (f" خلال {elapsed / 1000:.2f} ثانية" if elapsed else ""))
            return np.array(samples, dtype=np.float32), elapsed
        elif line.isdigit() or (line.startswith("-") and line[1:].isdigit()):
            samples.append(int(line))


def fix_rate(samples, elapsed_ms):
    """لو المعدل الفعلي يختلف عن 4000 عينة/ثانية بأكثر من 5% نعيد أخذ العينات على الزمن الحقيقي، حتى لا تتشوّه الترددات."""
    n = len(samples)
    if not elapsed_ms or n < 2:
        return samples
    actual = n / (elapsed_ms / 1000.0)
    if abs(actual - SAMPLE_RATE) / SAMPLE_RATE > 0.05:
        print(f"تنبيه: المعدل الفعلي {actual:.0f} عينة/ثانية (المطلوب {SAMPLE_RATE}). صحّحته بإعادة أخذ العينات.")
        dur = elapsed_ms / 1000.0
        t_old = np.linspace(0, dur, n, endpoint=False)
        t_new = np.arange(int(round(dur * SAMPLE_RATE))) / SAMPLE_RATE
        samples = np.interp(t_new, t_old, samples).astype(np.float32)
    return samples


def samples_to_wav(samples, path):
    # تحويل من مدى ADC (0-1023) إلى موجة موقّعة طبيعية
    centered = samples - np.mean(samples)
    normalized = centered / (np.max(np.abs(centered)) + 1e-9)
    audio_int16 = (normalized * 32767 * 0.9).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, audio_int16)


def send_to_server(wav_path, palm_id):
    with open(wav_path, "rb") as f:
        files = {"audio": ("recording.wav", f, "audio/wav")}
        data = {"palm_id": palm_id}
        resp = requests.post(SERVER_URL, files=files, data=data, timeout=30,
                             headers={"X-Nakhla-Key": DEVICE_KEY} if DEVICE_KEY else None)
    resp.raise_for_status()
    return resp.json()


def main():
    if len(sys.argv) < 2:
        print("الاستخدام: python serial_bridge.py <PORT>  مثال: COM3 أو /dev/ttyUSB0")
        sys.exit(1)

    port = sys.argv[1]
    ser = serial.Serial(port, BAUD, timeout=2)
    time.sleep(2)  # وقت إعادة تشغيل Arduino بعد فتح المنفذ

    # يحفظ آخر رقم نخلة استُخدم، عشان لو تفحصين نفس النخلة مرتين على التوالي
    # تضغطين Enter بس بدل ما تعيدين كتابة نفس الرقم من جديد
    last_palm_id = DEFAULT_PALM_ID

    while True:
        # لو نقلتِ الحساس لنخلة جديدة، اكتبي رقمها هنا قبل كل تسجيل.
        # لو هي نفس النخلة اللي فحصتيها قبل شوي، بس اضغطي Enter فاضي.
        entered = input(f"\nرقم النخلة الحالية [{last_palm_id}]: ").strip()
        palm_id = entered if entered else last_palm_id
        last_palm_id = palm_id
        ser.reset_input_buffer()
        ser.write(b"R")   # يطلب من الأردوينو بدء تسجيل (ما تحتاجين زرًا)

        samples, elapsed = read_one_recording(ser)
        samples = fix_rate(samples, elapsed)
        if len(samples) < SAMPLE_RATE:  # تسجيل ناقص/فاشل
            print("التسجيل قصير جدًا فتم تجاهله. حاوِل مرة أخرى.")
            continue

        wav_path = "recording_tmp.wav"
        samples_to_wav(samples, wav_path)
        print(f"جارٍ الإرسال إلى الخادم (النخلة: {palm_id})...")
        try:
            result = send_to_server(wav_path, palm_id)
            cls = result["fusion_result"]["final_class"]
            print(f"\n>>> النخلة: {palm_id}")
            print(f">>> النتيجة: {cls}")
            print(">>> التوصية:", result["recommendation"], "\n")
        except Exception as e:
            print("خطأ في الاتصال بالخادم (تأكد من أن app.py يعمل):", e)


if __name__ == "__main__":
    main()
