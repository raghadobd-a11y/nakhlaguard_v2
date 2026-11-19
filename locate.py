"""
تحديد العلامات الظاهرة للإصابة على صورة جذع النخلة (Gemini Vision) - مشروع نخلة
=================================================================================
⚠️ حدود هذه الميزة (مهمة):
- الصورة لا تُظهر اليرقات داخل الجذع. أقصى ما يمكن رصده هو علامات خارجية ظاهرة (فتحات، إفرازات، نشارة ألياف، تلف السعف).
- هذه العلامات قد تنتج عن أسباب أخرى، فالنتيجة ملاحظات أولية تُوجّه الفحص ولا تشخّص.
- لا تقترح الميزة مبيدًا ولا جرعة ولا مواضع حقن. قرار المعالجة والحقن لمختص زراعي.
"""
import os
import io
import json
import base64
import asyncio
from collections import Counter

ALLOWED_TYPES = {"exit_hole", "ooze", "frass", "fiber_damage", "crown_damage", "adult_weevil", "other"}
ALLOWED_CONF = {"low", "medium", "high"}
MAX_SIGNS = 8

LOCATE_PROMPT = """You are helping a farmer inspect a date palm. Look at the photo and mark ONLY external,
visible signs that are sometimes associated with red palm weevil damage. Typical signs:
- exit_hole: round or irregular holes in the trunk or at frond bases
- ooze: brown, sticky or dried liquid stains running down the trunk or frond base
- frass: chewed fibrous material that looks like sawdust or wet brown fibre
- fiber_damage: cut, chewed or torn fibres and frond bases
- crown_damage: collapsed, drooping or dry central fronds in the crown
- adult_weevil: an ADULT red palm weevil clearly visible on the surface (a large reddish-brown beetle, 2 to 4 cm, with a long curved snout) on the trunk, the crown or the ground next to the palm
- other: any other unusual damage worth inspecting

STRICT RULES:
- You cannot see larvae or anything inside the trunk. Never claim to see larvae and never say where they are. The only living thing you may mark is a visible ADULT weevil (type adult_weevil), and only when it is unmistakable; many other beetles look similar, so use low or medium confidence unless it is very clear.
- Do not recommend any pesticide, dose, injection point or treatment. Only say where a person should look closely.
- Each sign can have other causes. Be conservative: if you are unsure, give "low" confidence or leave it out.
- If the photo does not show a date palm clearly, set "palm_visible" to false and return no signs.
- Coordinates are normalized to the image: x from 0 (left) to 1 (right), y from 0 (top) to 1 (bottom).
- "trunk_box" is the bounding box of the visible trunk (x0, y0, x1, y1), or null if no trunk is visible.
- Keep every note under 15 words, written in {language_name}.

Respond with ONLY a JSON object, no other text, in exactly this format:
{"palm_visible": true or false,
 "photo_quality": "good" or "poor",
 "trunk_box": {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0} or null,
 "signs": [{"type": "exit_hole|ooze|frass|fiber_damage|crown_damage|adult_weevil|other", "x": 0.0, "y": 0.0, "confidence": "low|medium|high", "note": "<short note>"}],
 "summary": "<one or two sentences: what is visible and what to inspect, in {language_name}>"}"""

DISCLAIMER = {
    "ar": "هذه علامات ظاهرة تستحق الفحص، وليست مواضع اليرقات داخل الجذع، وقد تنتج عن أسباب أخرى. لا يغني هذا الفحص عن المرشد الزراعي، وقرار المعالجة والحقن والمبيد والجرعة يحدده مختص.",
    "en": "These are visible signs worth inspecting, not the position of larvae inside the trunk, and they can have other causes. This check does not replace an agricultural specialist, who decides on treatment, injection, pesticide and dose.",
    "ur": "یہ نظر آنے والی علامات ہیں جو معائنے کے لائق ہیں، تنے کے اندر لاروے کا مقام نہیں، اور ان کی دوسری وجوہات بھی ہو سکتی ہیں۔ یہ جانچ زرعی ماہر کا متبادل نہیں؛ علاج، انجکشن، کیڑے مار دوا اور اس کی مقدار کا فیصلہ ماہر کرتا ہے۔",
}

LANG_NAME = {"en": "English", "ar": "Modern Standard Arabic (simple, farmer-friendly)", "ur": "Urdu"}


def _clamp01(v):
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return None


def sanitize(parsed, language="ar"):
    """يتحقق من ناتج النموذج ويحصره بنطاق آمن، ولا يثق بأي حقل منه."""
    if not isinstance(parsed, dict):
        return None
    lang = language if language in ("en", "ar", "ur") else "ar"
    box = parsed.get("trunk_box")
    trunk_box = None
    if isinstance(box, dict):
        x0, y0, x1, y1 = (_clamp01(box.get(k)) for k in ("x0", "y0", "x1", "y1"))
        if None not in (x0, y0, x1, y1) and x1 - x0 > 0.05 and y1 - y0 > 0.05:
            trunk_box = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    signs = []
    for s in (parsed.get("signs") or [])[:MAX_SIGNS]:
        if not isinstance(s, dict):
            continue
        x, y = _clamp01(s.get("x")), _clamp01(s.get("y"))
        if x is None or y is None:
            continue
        typ = s.get("type") if s.get("type") in ALLOWED_TYPES else "other"
        conf = s.get("confidence") if s.get("confidence") in ALLOWED_CONF else "low"
        note = str(s.get("note") or "")[:140]
        # الموضع النسبي على الجذع: u عرضًا (0 يسار - 1 يمين)، v ارتفاعًا (0 أسفل - 1 أعلى)
        if trunk_box:
            u = (x - trunk_box["x0"]) / (trunk_box["x1"] - trunk_box["x0"])
            v = 1 - (y - trunk_box["y0"]) / (trunk_box["y1"] - trunk_box["y0"])
        else:
            u, v = x, 1 - y
        signs.append({"type": typ, "x": x, "y": y, "confidence": conf, "note": note,
                      "rel_u": max(0.0, min(1.0, u)), "rel_v": max(0.0, min(1.0, v)),
                      "outside_trunk": bool(trunk_box and not (0 <= u <= 1 and 0 <= v <= 1)),
                      "verified": None, "verify_note": ""})
    palm_visible = bool(parsed.get("palm_visible", True))
    if not palm_visible:
        signs = []
    return {
        "palm_visible": palm_visible,
        "photo_quality": "poor" if parsed.get("photo_quality") == "poor" else "good",
        "trunk_box": trunk_box,
        "signs": signs,
        "summary": str(parsed.get("summary") or "")[:500],
        "disclaimer": DISCLAIMER[lang],
    }


# ---------- فحص ثانٍ بتكبير: يقصّ منطقة صغيرة حول كل علامة ويسأل Gemini عنها بصرامة ----------
# يقلّل العلامات الخاطئة التي تنتج عن لقطة واسعة (لحاء، ظل، قاعدة سعف عادية). لا يثبت صحة الإصابة، ولم يُقَس أثره بعد (انظر eval_locate.py --verify).
MAX_VERIFY = 6
VERIFY_DESC = {"exit_hole": "round or irregular hole in the trunk or at a frond base", "ooze": "brown sticky or dried liquid stain", "frass": "chewed fibre or sawdust-like material",
               "fiber_damage": "cut, chewed or torn fibres at a frond base", "crown_damage": "collapsed, drooping or dry central fronds", "adult_weevil": "adult red palm weevil: a large reddish-brown beetle with a long curved snout", "other": "unusual damage"}
VERIFY_PROMPT = """You are double-checking ONE spot on a photo of a date palm. The image is a zoomed crop. A first look marked a POSSIBLE sign of type "{type}" ({desc}) at x={px}, y={py} of this crop (0,0 = top-left, 1,1 = bottom-right).

Look only at this crop and be STRICT:
- "yes": the sign is clearly visible at or very near that spot.
- "no": the spot shows only normal bark, shadow, a normal frond base, dirt or texture, or nothing unusual.
- "uncertain": the crop is too blurry, dark or small to tell.
You cannot see larvae inside the trunk; never mention them and never give treatment advice.
Respond with ONLY a JSON object, no other text: {"sign_visible": "yes" or "no" or "uncertain", "type": "exit_hole|ooze|frass|fiber_damage|crown_damage|adult_weevil|other|none", "note": "<max 12 words in {language_name}>"}"""


def _crop_around(image_bytes, x, y, frac=0.25, out_side=640):
    """يرجّع (jpeg_bytes, موضع العلامة x داخل القصة, y) أو None لو Pillow غير مثبّتة أو الصورة صغيرة/غير مقروءة."""
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    W, H = im.size
    side = int(frac * min(W, H))
    if side < 96:
        return None
    cx, cy = x * W, y * H
    x0 = int(max(0, min(W - side, cx - side / 2)))
    y0 = int(max(0, min(H - side, cy - side / 2)))
    buf = io.BytesIO()
    im.crop((x0, y0, x0 + side, y0 + side)).resize((out_side, out_side)).save(buf, format="JPEG", quality=88)
    return buf.getvalue(), (cx - x0) / side, (cy - y0) / side


async def _verify_one(image_bytes, sign, language, api_key, sem):
    from gemini_client import call_gemini
    cropped = _crop_around(image_bytes, sign["x"], sign["y"])
    if cropped is None:
        return None
    jpg, px, py = cropped
    prompt = (VERIFY_PROMPT.replace("{type}", sign["type"]).replace("{desc}", VERIFY_DESC.get(sign["type"], "unusual damage"))
              .replace("{px}", f"{px:.2f}").replace("{py}", f"{py:.2f}").replace("{language_name}", LANG_NAME.get(language, "Arabic")))
    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(jpg).decode("utf-8")}}]}]}
    async with sem:
        try:
            data = await call_gemini(payload, api_key)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            j = json.loads(text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip())
        except Exception as e:
            print(f"[NakhlaGuard] Zoom check failed: {type(e).__name__}: {e}")
            return None
    if not isinstance(j, dict):
        return None
    v = j.get("sign_visible")
    v = v if v in ("yes", "no", "uncertain") else "uncertain"
    t = j.get("type") if j.get("type") in ALLOWED_TYPES else "none"
    return {"verified": {"yes": "yes", "no": "no", "uncertain": "unclear"}[v], "type": t, "note": str(j.get("note") or "")[:120]}


async def _apply_verification(result, image_bytes, language, api_key):
    try:
        import PIL  # noqa: F401
    except ImportError:
        result["verification"] = {"requested": True, "available": False}
        return result
    signs = result["signs"][:MAX_VERIFY]
    sem = asyncio.Semaphore(2)
    outs = await asyncio.gather(*[_verify_one(image_bytes, sg, language, api_key, sem) for sg in signs])
    counts = Counter()
    for sg, o in zip(signs, outs):
        if o is None:
            counts["error"] += 1
            continue
        sg["verified"], sg["verify_note"] = o["verified"], o["note"]
        if o["verified"] == "yes" and o["type"] in ALLOWED_TYPES:
            sg["type"] = o["type"]          # القصة المكبّرة أدق من اللقطة الواسعة
        counts[o["verified"]] += 1
    result["verification"] = {"requested": True, "available": True, "checked": len(signs), "yes": counts["yes"], "no": counts["no"], "unclear": counts["unclear"], "error": counts["error"]}
    return result


async def locate_signs_from_image(image_bytes, mime_type="image/jpeg", language="ar", verify=False):
    """يرجّع dict منقّحة، أو None لو ما فيه مفتاح Gemini أو فشل الاتصال أو كان الرد غير مفهوم. verify=True يضيف الفحص الثاني بتكبير."""
    from gemini_client import call_gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    language_name = LANG_NAME.get(language, "Arabic")
    payload = {"contents": [{"role": "user", "parts": [
        {"text": LOCATE_PROMPT.replace("{language_name}", language_name)},
        {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode("utf-8")}},
    ]}]}
    try:
        data = await call_gemini(payload, api_key)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = sanitize(json.loads(text), language)
    except Exception as e:
        print(f"[NakhlaGuard] Locate-signs failed: {type(e).__name__}: {e}")
        return None
    if result is not None and verify and result["signs"]:
        result = await _apply_verification(result, image_bytes, language, api_key)
    return result
