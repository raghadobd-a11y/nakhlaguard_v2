"""
عميل مشترك للاتصال بـ Gemini API - يستخدمه fusion.py (التوصية/التقرير
الكامل) و agent.py (الوكيل الذكي) بدل ما كل ملف يكرر نفس منطق الاتصال.

Gemini أحيانًا يرجع:
- 503 (الخادم مشغول مؤقتًا - شائع مع النماذج الجديدة اللي عليها ضغط استخدام)
- 429 (تجاوزتِ حد الطلبات المسموح)

هذي أخطاء عابرة (Transient) تُحل غالبًا بإعادة المحاولة بعد ثانية أو
ثانيتين. لكن لو نموذج معين ضل مزدحم حتى بعد كل المحاولات (شائع مع
النماذج الجديدة جدًا زي gemini-3.8-flash وقت ضغط الاستخدام العالي)،
ننتقل تلقائيًا لنموذج احتياطي أخف وأقدم بدل ما نفشل الطلب كامل.

الاستثناء الوحيد: خطأ 401 (مفتاح غير صالح) يفشل بنفس الشكل مع أي
نموذج، فنرميه فورًا بدون إضاعة وقت بتجربة نماذج إضافية.
"""
import asyncio
import os
import time
import httpx

# نموذج أساسي جديد وقوي، ثم بديلان أقدم وأخف لو الأول مزدحم
# للتغيير بدون تعديل الكود: متغير البيئة GEMINI_MODELS، أسماء مفصولة بفواصل بالترتيب، مثل: gemini-3.5-flash-lite,gemini-3.8-flash
# (الأسرع أولًا يعطي إجابات أسرع، والأقوى أولًا يعطي نصًا أغنى). الافتراضي: الأقوى ثم الأسرع ثم الأخف.
# gemini-2.5-flash أُخرج من القائمة لأن Google أعلنت تقييد الوصول لنماذج 2.5 على من استعملها سابقًا فقط.
_ENV_MODELS = [m.strip() for m in os.environ.get("GEMINI_MODELS", "").split(",") if m.strip()]
FALLBACK_MODELS = _ENV_MODELS or ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

TRANSIENT_STATUS_CODES = {429, 503}

# حدود الانتظار: حتى لو Gemini بطيء أو محجوب، لا يعلّق الموقع. أطول انتظار كلي للطلب الواحد (بكل المحاولات
# والنماذج الاحتياطية) هو TOTAL_DEADLINE، وبعده يرجع الخطأ فيعرض الموقع النص البديل المكتوب سلفًا بدل الانتظار.
# للتعديل من غير تغيير الكود: متغيرات البيئة GEMINI_TOTAL_DEADLINE و GEMINI_REQUEST_TIMEOUT (بالثواني).
import os
TOTAL_DEADLINE = float(os.environ.get("GEMINI_TOTAL_DEADLINE", "35"))
REQUEST_TIMEOUT = float(os.environ.get("GEMINI_REQUEST_TIMEOUT", "20"))

# قاعدة صياغة تُضاف لكل طلب: نقول "نخلة" ولا نقول "شجرة" أبدًا
WORDING_RULE = ("Wording rule for every answer, in every language: always say 'palm' and never 'tree'. "
                "In Arabic write نخلة / نخيل / نخلات and never شجرة / أشجار. "
                "In Urdu write کھجور and never درخت. In English write 'palm' and never 'tree'.")


def _with_wording_rule(payload):
    """نسخة سطحية من الطلب مع إضافة قاعدة الصياغة لتعليمات النظام (بدون نسخ الصور الكبيرة)."""
    out = dict(payload)
    si = dict(out.get("system_instruction") or {})
    parts = list(si.get("parts") or [])
    parts.append({"text": WORDING_RULE})
    si["parts"] = parts
    out["system_instruction"] = si
    return out


async def _attempt_model(client, model_name, payload, headers, max_retries, base_delay):
    """
    يحاول نموذج واحد بعدد محاولات محدد (إعادة محاولة عند ازدحام مؤقت).
    يرجّع (data, None) عند النجاح، أو (None, الخطأ_الأخير) لو فشلت كل
    محاولات هذا النموذج تحديدًا - عندها الطالب ينتقل للنموذج التالي.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            used_param = False
            if resp.status_code == 401:
                # بعض أنواع المفاتيح (مثل التي تبدأ بـ AQ.) تُقبل كمعامل في الرابط ولا تُقبل كترويسة: نجرّب الطريقة الثانية
                key = headers.get("x-goog-api-key")
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"}, params={"key": key})
                used_param = True
                if resp.status_code == 401:
                    raise RuntimeError("401 Unauthorized: Google rejected the API key (tried it as a header and as a URL parameter)")
            if used_param and resp.status_code >= 400:
                # لا نمرّر الخطأ الأصلي لأن رسالته تحتوي الرابط وفيه المفتاح
                raise httpx.HTTPStatusError(f"{resp.status_code} from Gemini (key sent as URL parameter)", request=resp.request, response=resp)
            resp.raise_for_status()
            return resp.json(), None
        except httpx.TimeoutException as e:
            # ما رد النموذج بالوقت المحدد (ReadTimeout وأخواتها): ما نعيد على نفس النموذج، ننتقل فورًا للتالي
            print(f"[NakhlaGuard] {model_name} -> timeout ({type(e).__name__}), trying next model")
            last_error = e
            break
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status == 401:
                raise  # مشكلة بالمفتاح نفسه - ما تتحل بتغيير النموذج
            last_error = e
            if status in TRANSIENT_STATUS_CODES and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[NakhlaGuard] {model_name} -> {status}, retrying in {delay:.1f}s")
                await asyncio.sleep(delay)
                continue
            break  # خطأ غير عابر، أو استنفدنا محاولات هذا النموذج تحديدًا
    return None, last_error


async def call_gemini(payload, api_key, models=None, max_retries=2, base_delay=1.5):
    """
    يجرّب كل نموذج بقائمة models بالترتيب (افتراضيًا FALLBACK_MODELS).
    لو نموذج فشل تمامًا بعد كل محاولاته (503/429 مستمر)، ينتقل تلقائيًا
    للنموذج التالي بالقائمة - يرفع فرصة نجاح الطلب بشكل كبير وقت ضغط
    الاستخدام العالي على نموذج معين (شي متوقع مع نموذج جديد جدًا).
    """
    models = models or FALLBACK_MODELS
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    last_error = None
    payload = _with_wording_rule(payload)
    deadline = time.monotonic() + TOTAL_DEADLINE

    async with httpx.AsyncClient(timeout=httpx.Timeout(REQUEST_TIMEOUT, connect=8.0)) as client:
        for model_name in models:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                data, error = await asyncio.wait_for(
                    _attempt_model(client, model_name, payload, headers, max_retries, base_delay), timeout=remaining)
            except asyncio.TimeoutError:
                print(f"[NakhlaGuard] total deadline ({TOTAL_DEADLINE:.0f}s) reached on {model_name}")
                last_error = asyncio.TimeoutError(f"Gemini did not answer within {TOTAL_DEADLINE:.0f}s")
                break
            if data is not None:
                return data
            print(f"[NakhlaGuard] {model_name} exhausted - trying next fallback model")
            last_error = error

    raise last_error or asyncio.TimeoutError(f"Gemini did not answer within {TOTAL_DEADLINE:.0f}s")
