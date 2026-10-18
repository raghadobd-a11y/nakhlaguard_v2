"""
معرفة عمر النخلة وتأثيره على قابلية الإصابة بسوسة النخيل الحمراء.
مبنية على معلومات زراعية حقيقية (مو مولّدة) قدّمتها المستخدمة، تُستخدم
لإثراء تقرير الذكاء الاصطناعي والوكيل بسياق إضافي مبني على عمر النخلة،
مو بس على الصوت وحده.
"""


def age_risk_bracket(age_years):
    """
    يرجّع (فئة_الخطورة, وصف_إنجليزي, وصف_عربي) حسب عمر النخلة بالسنوات.
    فئات الخطورة: "high", "moderate", "low" - تُستخدم كسياق إضافي فقط،
    ما تغيّر تصنيف النموذج الصوتي نفسه (اللي يبقى المصدر الأساسي للقرار).
    """
    if age_years is None:
        return None

    if age_years < 3:
        return (
            "high",
            "Newly planted offshoot (under 3 years). The fresh wound where the "
            "offshoot was cut from the mother palm releases sap that strongly "
            "attracts egg-laying females - this is a known high-risk window.",
            "فسيلة حديثة الغرس (أقل من 3 سنوات). جرح فصلها عن النخلة الأم "
            "يفرز عصارة تجذب إناث السوسة لوضع البيض بقوة - هذي فترة خطورة معروفة.",
        )
    if 3 <= age_years <= 15:
        return (
            "high",
            "Young palm (3-15 years, peak risk at 5-10). Trunk tissue is still "
            "soft, moist, and sugar-rich - ideal food for larvae - and ground-level "
            "offshoots and frond bases provide easy hiding spots for egg-laying.",
            "نخلة شابة (3-15 سنة، وذروة الخطورة بين 5-10 سنوات). أنسجة الجذع "
            "لسا طرية ورطبة وغنية بالسكريات - غذاء مثالي لليرقات - والفسائل "
            "وقواعد السعف القريبة من الأرض توفر مخابئ سهلة لوضع البيض.",
        )
    if 15 < age_years <= 20:
        return (
            "moderate",
            "Transitional age (15-20 years). Risk is decreasing as trunk wood "
            "begins to harden, but the palm is not yet in the lower-risk bracket.",
            "عمر انتقالي (15-20 سنة). الخطورة تبدأ تنخفض مع بداية تصلّب خشب "
            "الجذع، لكن النخلة لسا ما دخلت فئة الخطورة المنخفضة كاملة.",
        )
    return (
        "low",
        "Mature palm (over 20 years). Trunk wood is hard and lignified, making it "
        "harder for larvae to bore in - infestation risk is lower. If it does "
        "occur, it typically targets the growing crown (jummar) rather than the "
        "trunk base, since ground-level tissue is now too hard for easy entry.",
        "نخلة متقدمة بالعمر (أكثر من 20 سنة). خشب الجذع صلب ومتلجّن، فيصعب على "
        "اليرقات اختراقه - احتمال الإصابة أقل. لو حصلت، غالبًا تستهدف الجُمّارة "
        "(القمة النامية) بدل قاعدة الجذع، لأن الأنسجة القريبة من الأرض صارت "
        "صلبة يصعب اختراقها بسهولة.",
    )


def age_context_text(age_years, language="en"):
    """يبني سطر سياق جاهز يُضاف لبرومبت Gemini، أو None لو ما فيه عمر مسجّل."""
    result = age_risk_bracket(age_years)
    if result is None:
        return None
    risk_level, text_en, text_ar = result
    text = text_ar if language == "ar" else text_en
    label = {"high": "HIGH", "moderate": "MODERATE", "low": "LOWER"}[risk_level]
    if language == "ar":
        label = {"high": "مرتفعة", "moderate": "متوسطة", "low": "منخفضة"}[risk_level]
        return f"عمر النخلة: {age_years} سنة (فئة خطورة عمرية: {label}) - {text}"
    return f"Palm age: {age_years} years (age-related risk bracket: {label}) - {text}"
