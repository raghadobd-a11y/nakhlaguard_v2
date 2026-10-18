"""
دمج النتائج (Fusion) + التوصية الذكية - مشروع "نخلة"
===================================================
هذا الملف يمثّل نقطة التميّز الأساسية بالمشروع (الأصالة الحقيقية):
الأبحاث المنشورة تتوقف عند "تصنيف الإشارة الصوتية" كنتيجة أكاديمية.
هنا نضيف طبقتين ما وجدناهما بالأبحاث المنشورة:

1. دمج الصوت + الصورة (Multi-modal Fusion) لرفع الثقة بالقرار
2. تحويل التصنيف التقني لتوصية عملية + تقدير خسارة اقتصادية عبر Gemini،
   بما يحوّل "نتيجة نموذج" إلى "قرار يفهمه المزارع مباشرة"
"""
import os
import json
import base64
from severity_labels import anglicize

SEVERITY_WEIGHT = {"سليم": 0.0, "مبكرة": 0.33, "متوسطة": 0.66, "متقدمة": 1.0}
SEVERITY_ORDER = ["سليم", "مبكرة", "متوسطة", "متقدمة"]


# ---------- تقدير العمر وفحص بصري عام لحالة النخلة من صورة (Gemini Vision) ----------
# ⚠️ هذا فحص بصري أولي (Pre-screening) فقط، مو تشخيص طبي/مرضي. أمراض زي
# الوجام (Phytoplasma) أو الخامج تحتاج تأكيدًا مخبريًا حقيقيًا - الكاميرا
# بس تلاحظ علامات ظاهرة عامة وتنبّه المزارع يراجع مختص، ولا تجزم بأي مرض.
AGE_HEALTH_PROMPT = """You are looking at a photo of a date palm tree. Based on \
visual cues only, give your best APPROXIMATE estimate for each item below.

For age: use trunk height/thickness, frond crown density, presence of ground-level offshoots.
For each visual flag: look only for the specific sign described - do not diagnose any disease,
since symptoms like yellowing can have many causes (pests, disease, salinity, nutrient deficiency)
that visually overlap and cannot be told apart from a photo alone.

Respond with ONLY a JSON object, no other text, in exactly this format:
{"estimated_age_years": <a single integer, your best middle estimate>,
 "age_confidence": "low" or "medium",
 "drought_stress_visible": true or false,
 "drought_notes": "<one short sentence on frond wilting/dryness, or empty string>",
 "dust_webbing_visible": true or false,
 "dust_webbing_notes": "<one short sentence on dusty webbing on fruit bunches, or empty string>",
 "leaf_yellowing_visible": true or false,
 "leaf_yellowing_notes": "<one short sentence on yellow streaking or leaf-tip burn, or empty string>",
 "frond_base_rot_visible": true or false,
 "frond_base_rot_notes": "<one short sentence on rot/dieback at frond or offshoot bases, or empty string>"}

Be conservative and honest - this is a rough visual estimate, not a precise \
measurement or a diagnosis. If you cannot see the palm clearly, still give your \
best guess with "age_confidence": "low", and leave flags false if unsure."""


async def estimate_age_and_health_from_image(image_bytes, mime_type="image/jpeg"):
    """
    يرسل صورة النخلة لـ Gemini Vision ويرجّع تقدير تقريبي لعمرها + هل فيه
    علامات إجهاد/جفاف ظاهرة - كل هذا بدون أي حساس إضافي، بس من نفس الصورة
    اللي أصلاً يرفعها المزارع لتحليل الإصابة.

    يرجّع dict فيها estimated_age_years, age_confidence، وأربعة أعلام بصرية
    (drought/dust_webbing/leaf_yellowing/frond_base_rot) كل وحدة بصيغة
    <flag>_visible + <flag>_notes - أو None لو ما فيه مفتاح Gemini أو صار
    خطأ بالاتصال. كلها ملاحظات بصرية أولية فقط، مو تشخيصًا طبيًا.
    """
    from gemini_client import call_gemini

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": AGE_HEALTH_PROMPT},
                {"inline_data": {"mime_type": mime_type, "data": b64_image}},
            ],
        }],
    }
    try:
        data = await call_gemini(payload, api_key)
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except Exception as e:
        print(f"[NakhlaGuard] Age/health image estimate failed: {type(e).__name__}: {e}")
        return None


def fuse_audio_image(audio_result, image_infested_probability=None, image_weight=0.35):
    """
    يدمج نتيجة النموذج الصوتي مع نتيجة تحليل الصورة (لو متوفرة).

    audio_result: مخرجات predict_severity_from_array()
    image_infested_probability: احتمال وجود إصابة من تحليل الصورة (0-1)،
        عادة تجيء من استدعاء Gemini Vision على صورة الجذع (أعراض ظاهرية:
        ثقوب، إفرازات صمغية، نشارة خشب). لو ما فيه صورة، يتجاهل الدمج
        ويعتمد على الصوت فقط.
    """
    audio_score = sum(
        SEVERITY_WEIGHT[cls] * p for cls, p in audio_result["probabilities"].items()
    )

    if image_infested_probability is None:
        final_score = audio_score
        used_fusion = False
    else:
        final_score = (1 - image_weight) * audio_score + image_weight * image_infested_probability
        used_fusion = True

    # تحويل النتيجة المدمجة (0-1) رجوع لأقرب فئة شدة
    idx = min(int(round(final_score * (len(SEVERITY_ORDER) - 1))), len(SEVERITY_ORDER) - 1)
    final_class = SEVERITY_ORDER[idx]

    return {
        "audio_only_score": round(audio_score, 3),
        "fusion_used": used_fusion,
        "final_score": round(final_score, 3),
        "final_class": final_class,
    }


NEEDS_ATTENTION = {"متوسطة", "متقدمة"}

# النظام يدعم لغتين الآن - المستخدم يختار (زر تبديل باللغة بالواجهة)
# ويُرسل معه "language" ("en" أو "ar") مع كل طلب، فيولّد Gemini جوابه
# بنفس اللغة المطلوبة بدل ما تكون إنجليزي ثابت دائمًا.
SHORT_SYSTEM_PROMPT_EN = """You are an agricultural assistant specialized in Red Palm \
Weevil infestations in Saudi Arabia.
The palm is in a relatively good condition (healthy or very early infestation). \
Give a short note only (3-4 lines): reassure the farmer, and mention one simple \
preventive monitoring tip. If an "age_risk_context" or "visual_health_notes" \
field is present in the case data, briefly weave that into your note - phrase any \
visual health note as an observation to confirm with an agricultural expert, never as a diagnosis.
Respond in English only, plain and clear, with no preamble."""

FULL_REPORT_SYSTEM_PROMPT_EN = """You are an agricultural expert specialized in Red \
Palm Weevil infestations in Saudi Arabia.
The palm's condition requires real attention. Write a complete, organized report \
for the farmer using exactly the following section headers (use ## before each one):

## Diagnosis
A short paragraph explaining what the current classification means in plain terms.

## Treatment Plan
Clear, numbered, practical steps in the correct chronological order (first to last).

## Suggested Timeline
Write a 4-week plan. For each week write one line in the format: Week 1 | short focus title, followed by exactly 3 lines for that week's most important days only (skip the rest), each in the format: Day 1 | one short concrete task (max 10 words). Pick only the 3 days per week that matter most (e.g. the day to act, a mid-week check, and the review day) - do not list all 7. Repeat for Weeks 1 to 4 (16 lines in total). No other text in this section.

## Economic Estimate
A simple, realistic (relative, not an exact figure) comparison between the cost of \
treating now versus the expected loss if treatment is delayed - in a way that \
shows why early intervention is cheaper.

## Expected Harvest Impact
Estimate the impact on this season's date (rutab) yield if left untreated. Use \
40-80 kg as the typical seasonal yield range for a well-cared mature date palm \
(higher in well-irrigated regions like Qassim). Give an approximate percentage \
and kg range at risk based on severity (roughly 20-40% for moderate infestation, \
60-100% including possible tree loss for advanced infestation). Clearly label \
this as an illustrative estimate, not an exact measurement.

## Final Note
One motivating, realistic closing sentence.

If the case data includes an "age_risk_context" field, mention it briefly in \
the Diagnosis section (it explains WHY this palm's age makes it more or less \
vulnerable). If it includes "visual_health_notes", add one short line about it in the \
Diagnosis section too: state it as an observed visual sign only (not a confirmed \
diagnosis), recommend the farmer have it checked by an agricultural extension \
officer, and make clear it is separate from the weevil infestation finding.

Respond in English only, plain and clear, with no preamble before the first heading."""

SHORT_SYSTEM_PROMPT_AR = """أنت مساعد زراعي متخصص بآفة سوسة النخيل الحمراء بالسعودية.
النخلة بحالة جيدة نسبيًا (سليمة أو إصابة مبكرة جدًا). أعطِ ملاحظة قصيرة
فقط (3-4 أسطر): طمّن المزارع، واذكر نصيحة مراقبة وقائية بسيطة واحدة. لو
فيه حقل "سياق_خطورة_العمر" أو "ملاحظات_بصرية_صحية" ببيانات الحالة، اذكره
باختصار - أي ملاحظة بصرية صحية تُصاغ كعلامة ظاهرة للمتابعة مع مختص، لا كتشخيص.
أجب بالعربية الفصحى المبسطة فقط، بوضوح، بدون أي مقدمة."""

FULL_REPORT_SYSTEM_PROMPT_AR = """أنت خبير زراعي متخصص بآفة سوسة النخيل الحمراء بالسعودية.
حالة النخلة تستدعي عناية فعلية. اكتب تقريرًا كاملاً ومنظّمًا للمزارع
بنفس العناوين التالية بالضبط (استخدمي ## قبل كل عنوان):

## التشخيص
فقرة قصيرة تشرح معنى التصنيف الحالي بلغة بسيطة يفهمها المزارع.

## خطة العلاج
خطوات عملية مرقّمة وواضحة، بالترتيب الزمني الصحيح (من الأولى للأخيرة).

## الجدول الزمني المقترح
اكتب خطة لمدة 4 أسابيع. لكل أسبوع اكتب سطرًا بهذا الشكل: الأسبوع 1 | عنوان قصير للمحور، ثم 3 أسطر بس لأهم أيام ذلك الأسبوع (تجاهل الباقي)، بهذا الشكل: اليوم 1 | مهمة عملية قصيرة واحدة (بحد أقصى 10 كلمات). اختاري بس 3 أيام بكل أسبوع الأهم (مثل: يوم التنفيذ، فحص منتصف الأسبوع، يوم المراجعة) - لا تسردي كل الأيام السبعة. كرري ذلك للأسابيع من 1 إلى 4 (16 سطرًا بالمجموع). بدون أي نص آخر في هذا القسم.

## التقدير الاقتصادي
مقارنة بسيطة وواقعية (نسبية لا رقمًا دقيقًا) بين تكلفة العلاج الآن والخسارة
المتوقعة لو تأخر العلاج - بأسلوب يوضح لماذا التدخل المبكر أوفر.

## الأثر المتوقع على المحصول
قدّر أثر الإصابة على محصول الرطب لهذا الموسم لو ما تم العلاج. استخدم
40-80 كيلوجرام كنطاق إنتاج موسمي معتاد لنخلة ناضجة بعناية جيدة (أعلى
بمناطق الري الجيد زي القصيم). أعطِ نسبة تقريبية ومدى بالكيلوجرام معرّض
للخطر حسب شدة الإصابة (تقريبًا 20-40% للإصابة المتوسطة، و60-100% مع
احتمال فقدان النخلة كاملة للإصابة المتقدمة). وضّح بصراحة إنه تقدير
توضيحي تقريبي، مو قياس دقيق.

## نصيحة أخيرة
جملة واحدة تحفيزية وواقعية.

لو بيانات الحالة فيها حقل "سياق_خطورة_العمر"، اذكره باختصار بقسم
التشخيص (يشرح ليش عمر هذي النخلة يزيد أو يقلل قابليتها للإصابة). لو فيها
حقل "ملاحظات_بصرية_صحية"، أضف سطرًا قصيرًا عنه بقسم التشخيص كمان: اذكرها
كعلامة بصرية ملاحَظة فقط (مو تشخيصًا مؤكدًا)، وأوصِ بمراجعة مرشد زراعي
لتأكيدها، ووضّح إنها منفصلة عن نتيجة إصابة السوسة ولا ينبغي الخلط بينها.

أجب بالعربية الفصحى المبسطة فقط، بدون أي مقدمة قبل العنوان الأول."""


def build_gemini_prompt(fusion_result, palm_id=None, history=None, location=None, language="en",
                         age_years=None, visual_notes=None):
    """
    يبني الرسالة المُرسلة لـ Gemini API. يختار مستوى التفصيل تلقائيًا
    (ملاحظة قصيرة أو تقرير كامل)، ولغة الجواب حسب ما يطلبه المستخدم
    (language: "en" أو "ar").

    age_years: عمر النخلة (لو مسجّل يدويًا أو مُقدَّر من الصورة) - يضيف
    سياق خطورة عمرية حقيقي (شوفي age_risk.py) للتقرير.
    visual_notes: ملاحظات بصرية أولية مجمّعة من نفس صورة الإصابة (جفاف،
    خيوط ترابية، اصفرار سعف، تعفن قواعد) - فحص أولي مو تشخيصًا.

    بالإنجليزي: نترجم القيم الداخلية (anglicize) عشان الجواب ما يتسرب
    فيه أي كلمة عربية. بالعربي: نُبقي القيم كما هي (عربية أصلاً) لأن
    هذا أطبع لنموذج يجاوب بالعربي.
    """
    severity_value = (
        anglicize(fusion_result["final_class"]) if language == "en" else fusion_result["final_class"]
    )
    context = {
        "severity_class" if language == "en" else "شدة_الإصابة": severity_value,
        "risk_score" if language == "en" else "نسبة_الخطورة": fusion_result["final_score"],
    }
    if palm_id:
        context["palm_id" if language == "en" else "رقم_النخلة"] = palm_id
    if location:
        context["palm_location" if language == "en" else "موقع_النخلة"] = location
    if history:
        hist_value = anglicize(history) if language == "en" else history
        context["check_history" if language == "en" else "السجل_التاريخي"] = hist_value

    from age_risk import age_context_text
    age_note = age_context_text(age_years, language)
    if age_note:
        context["age_risk_context" if language == "en" else "سياق_خطورة_العمر"] = age_note
    if visual_notes:
        context["visual_health_notes" if language == "en" else "ملاحظات_بصرية_صحية"] = visual_notes

    needs_full_report = fusion_result["final_class"] in NEEDS_ATTENTION
    if language == "ar":
        system_prompt = FULL_REPORT_SYSTEM_PROMPT_AR if needs_full_report else SHORT_SYSTEM_PROMPT_AR
        intro = "حلّل الحالة التالية:\n"
    else:
        system_prompt = FULL_REPORT_SYSTEM_PROMPT_EN if needs_full_report else SHORT_SYSTEM_PROMPT_EN
        intro = "Analyze the following case:\n"

    user_message = intro + json.dumps(context, ensure_ascii=False, indent=2)
    return system_prompt, user_message


# الاستدعاء الفعلي لـ Gemini (نفس أسلوب بصيرة) - عبر عميل مشترك
# (gemini_client.py) يعيد المحاولة تلقائيًا عند ازدحام الخادم (503/429)
async def get_gemini_recommendation(fusion_result, palm_id=None, history=None, location=None, language="en",
                                     age_years=None, visual_notes=None):
    from gemini_client import call_gemini

    if language == "ar":
        fallback_text = (
            f"الحالة المصنّفة ({fusion_result['final_class']}) تستدعي "
            "فحصًا ميدانيًا لتأكيد النتيجة قبل اتخاذ قرار العلاج."
        )
        no_key_msg = "⚠️ لم يتم ضبط GEMINI_API_KEY - هذا نص توصية تجريبي بديل: "
        fail_msg = "⚠️ تعذّر الاتصال بـ Gemini حاليًا - هذا نص توصية تجريبي بديل: "
    else:
        fallback_text = (
            f"The classified condition ({anglicize(fusion_result['final_class'])}) "
            "requires a field inspection to confirm the result before deciding on treatment."
        )
        no_key_msg = "⚠️ GEMINI_API_KEY is not set - this is a placeholder recommendation: "
        fail_msg = "⚠️ Could not reach Gemini right now - this is a placeholder recommendation: "

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return no_key_msg + fallback_text

    system_prompt, user_message = build_gemini_prompt(
        fusion_result, palm_id, history, location, language, age_years, visual_notes
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
    }
    try:
        data = await call_gemini(payload, api_key)
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        # نطبع الخطأ الحقيقي بالترمنال عشان يسهل تشخيصه، ونرجّع نص بديل
        # بدل ما نفشل الطلب كامل (500 Internal Server Error)
        print(f"[NakhlaGuard] Gemini call failed: {type(e).__name__}: {e}")
        return fail_msg + fallback_text
