"""
يصنع باركود «موقع المالك» لعلبة الجهاز.

    python make_owner_qr.py --base-url https://اسم-موقعك.onrender.com
    python make_owner_qr.py --base-url https://اسم-موقعك.onrender.com --key <مفتاح-موجود>

- بدون --key يولّد مفتاحًا عشوائيًا جديدًا (32 حرفًا) ويطبعه.
- يكتب صورة owner_qr_<معرّف>.png جاهزة للطباعة، ويطبع السطر الذي تضعه في متغير البيئة على Render.
- لكل علبة مفتاح مختلف. أضف كل المفاتيح في NAKHLA_DEVICE_KEYS مفصولة بفاصلة.
- المفتاح سرّ: لا ترفعه على GitHub ولا تلصقه في دردشة أو صورة منشورة.
يحتاج مكتبة opencv (cv2) أو qrcode (pip install qrcode[pil])، ويستخدم الموجودة منهما.
"""
import argparse
import secrets
import sys


def make_qr_matrix(text):
    try:
        import cv2
        import numpy as np
        enc = cv2.QRCodeEncoder.create()
        img = enc.encode(text)
        return (np.asarray(img) > 127)
    except Exception:
        pass
    try:
        import qrcode
        qr = qrcode.QRCode(border=0, error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(text)
        qr.make(fit=True)
        import numpy as np
        return ~np.array(qr.get_matrix(), dtype=bool)          # True = خلية فاتحة (نفس اتفاق cv2)
    except Exception:
        sys.exit("ثبّت إحدى المكتبتين: pip install opencv-python  أو  pip install qrcode[pil]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="عنوان الموقع، مثل https://nakhlaguard.onrender.com")
    ap.add_argument("--key", default="", help="مفتاح موجود (اختياري). بدونه يُولَّد مفتاح جديد")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    key = a.key or secrets.token_urlsafe(24)
    if len(key) < 16:
        sys.exit("المفتاح قصير: 16 حرفًا على الأقل")
    base = a.base_url.rstrip("/")
    if not base.startswith("https://") and "localhost" not in base and "127.0.0.1" not in base:
        print("تنبيه: الأفضل أن يبدأ العنوان بـ https://")
    url = f"{base}/o/{key}"

    import numpy as np
    from PIL import Image, ImageDraw
    m = make_qr_matrix(url)                          # True = خلية فاتحة
    # بعض المشفّرات تضيف إطارًا أبيض؛ نقصّه ثم نضيف هامشًا ثابتًا
    dark = ~m
    ys, xs = np.where(dark)
    dark = dark[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    n = dark.shape[0]
    scale = max(8, 640 // (n + 8))
    size = (n + 8) * scale
    img = Image.new("L", (size, size + 70), 255)
    px = img.load()
    for r in range(n):
        for c in range(n):
            if dark[r, c]:
                for dy in range(scale):
                    for dx in range(scale):
                        px[(c + 4) * scale + dx, (r + 4) * scale + dy] = 0
    d = ImageDraw.Draw(img)
    d.text((16, size + 8), "NakhlaGuard - scan to open your site", fill=0)
    d.text((16, size + 28), "Device key: keep it private", fill=0)
    out = a.out or f"owner_qr_{key[:6]}.png"
    img.save(out)
    print("\nالمفتاح:", key)
    print("الرابط داخل الباركود:", url)
    print("الصورة:", out)
    print("\nضع هذا في متغيرات البيئة على Render (أضف المفاتيح الأخرى بفاصلة):")
    print("NAKHLA_DEVICE_KEYS=" + key)
    print("\nولجسر الأردوينو على جهازك:")
    print("  Windows:  set NAKHLA_KEY=" + key)
    print("  Mac/Linux: export NAKHLA_KEY=" + key)


if __name__ == "__main__":
    main()
