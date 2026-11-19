"""
Ready-made answers for the public assistant demo, used ONLY when the AI (Gemini) cannot answer in time: no key, a timeout, a busy server, a daily limit.

Why it exists: in front of judges the assistant must never sit there for half a minute and then show a red error. If the AI does not answer in time,
demo_agent.demo_chat returns one of these answers instead. The farm answers are built by plain code from the same 5-palm sample farm the AI reads
(the same tool functions), so the numbers are the demo farm's. The two general answers are short fixed texts. NONE of them is written by the AI,
and every one of them says so in its first line.

It understands the five suggested questions: which palm needs attention, which are due for a re-check, tell me about a palm ID, how to protect
neighbouring palms, what to watch for when pruning. Anything else gets the list of the farm's palms and a polite note.
"""
import re
from datetime import datetime

NOTE = {
    "ar": "ملاحظة: الذكاء الاصطناعي غير متاح الآن (مشغول أو وصل حدّه المسموح)، فهذه إجابة جاهزة وليست من كتابة حارس.",
    "en": "Note: the AI is not available right now (busy or at its limit), so this is a ready-made answer, not written by Haris.",
    "ur": "نوٹ: AI ابھی دستیاب نہیں (مصروف ہے یا اپنی حد تک پہنچ گیا)، اس لیے یہ تیار جواب ہے، حارس نے نہیں لکھا۔",
}
STATUS = {
    "ar": {"سليم": "سليمة", "مبكرة": "إصابة مبكرة", "متوسطة": "إصابة متوسطة", "متقدمة": "إصابة متقدمة"},
    "en": {"سليم": "healthy", "مبكرة": "early infestation", "متوسطة": "moderate infestation", "متقدمة": "advanced infestation"},
    "ur": {"سليم": "صحت مند", "مبكرة": "ابتدائی انفیکشن", "متوسطة": "درمیانی انفیکشن", "متقدمة": "شدید انفیکشن"},
}


def _ago(d, lang):
    if lang == "ar":
        if d == 0:
            return "اليوم"
        if d == 1:
            return "أمس"
        if d == 2:
            return "قبل يومين"
        return f"قبل {d} أيام" if d <= 10 else f"قبل {d} يومًا"
    if lang == "ur":
        return "آج" if d == 0 else "کل" if d == 1 else f"{d} دن پہلے"
    return "today" if d == 0 else "yesterday" if d == 1 else f"{d} days ago"


TXT = {
    "ar": {"none_att": "لا توجد نخلات تحتاج انتباهًا الآن في مزرعة العرض.", "att": "النخيل التي تحتاج انتباهًا الآن:",
           "none_due": "لا توجد نخيل متأخر فحصها (أكثر من 30 يومًا).", "due": "النخيل المتأخر فحصها (أكثر من 30 يومًا):",
           "line": lambda p, s, a: f"- {p}: الحالة {s}، آخر فحص {a}", "dueline": lambda p, a: f"- {p}: آخر فحص {a}",
           "notfound": "لا توجد نخلة بهذا الرقم في مزرعة العرض.", "all": "هذه نخيل مزرعة العرض:",
           "det": lambda p, loc, age: f"{p}: الموقع {loc}، العمر {age} سنوات.", "notes": "ملاحظات: ", "hist": "آخر الفحوصات:",
           "histline": lambda s, a: f"- {s} ({a})",
           "other": "لا أستطيع الإجابة عن هذا السؤال بدون الذكاء الاصطناعي. جرّب أحد الأسئلة المقترحة أو أعد المحاولة بعد قليل."},
    "en": {"none_att": "No palm on the demo farm needs attention right now.", "att": "Palms that need attention now:",
           "none_due": "No palm is overdue for a check (more than 30 days).", "due": "Palms due for a re-check (more than 30 days):",
           "line": lambda p, s, a: f"- {p}: {s}, last checked {a}", "dueline": lambda p, a: f"- {p}: last checked {a}",
           "notfound": "No palm with this ID exists on the demo farm.", "all": "These are the palms of the demo farm:",
           "det": lambda p, loc, age: f"{p}: location {loc}, age {age} years.", "notes": "Notes: ", "hist": "Latest checks:",
           "histline": lambda s, a: f"- {s} ({a})",
           "other": "I cannot answer this question without the AI. Try one of the suggested questions, or try again shortly."},
    "ur": {"none_att": "ڈیمو فارم میں اس وقت کسی کھجور کو توجہ درکار نہیں۔", "att": "جن کھجوروں کو ابھی توجہ چاہیے:",
           "none_due": "کوئی کھجور جانچ میں تاخیر کا شکار نہیں (30 دن سے زیادہ)۔", "due": "جن کھجوروں کی دوبارہ جانچ باقی ہے (30 دن سے زیادہ):",
           "line": lambda p, s, a: f"- {p}: {s}، آخری جانچ {a}", "dueline": lambda p, a: f"- {p}: آخری جانچ {a}",
           "notfound": "ڈیمو فارم میں اس نمبر کی کوئی کھجور نہیں۔", "all": "یہ ڈیمو فارم کی کھجوریں ہیں:",
           "det": lambda p, loc, age: f"{p}: مقام {loc}، عمر {age} سال۔", "notes": "نوٹس: ", "hist": "آخری جانچیں:",
           "histline": lambda s, a: f"- {s} ({a})",
           "other": "AI کے بغیر میں اس سوال کا جواب نہیں دے سکتا۔ تجویز کردہ سوالوں میں سے کوئی آزمائیں یا تھوڑی دیر بعد دوبارہ کوشش کریں۔"},
}

# General advice for the two knowledge chips. Short, cautious, and always pointing to the local agriculture authority.
GENERAL = {
    "protect": {
        "ar": ["إرشاد عام لحماية النخيل المجاورة لنخلة مصابة:",
               "- أبلغ جهة الزراعة المختصة بالنخلة المصابة، ولا تعالجها أو تقطعها قبل توجيهاتهم.",
               "- لا تنقل فسائل أو سعفًا أو مخلفات من النخلة المصابة إلى مكان آخر.",
               "- افحص النخيل المجاورة أكثر من المعتاد، خصوصًا قاعدة الجذع والقلب.",
               "- تجنّب جرح النخيل المجاورة (تقليم أو فصل فسائل) قدر الإمكان، فالجروح الطازجة تجذب السوس.",
               "- سجّل كل فحص في النظام لتتابع تغيّر الحالة."],
        "en": ["General guidance for protecting palms next to an infested one:",
               "- Report the infested palm to the local agriculture authority, and do not treat or cut it before they advise you.",
               "- Do not move offshoots, fronds or waste from the infested palm to another place.",
               "- Inspect the neighbouring palms more often than usual, especially the trunk base and the crown.",
               "- Avoid wounding the neighbouring palms (pruning, removing offshoots) where you can, because fresh wounds attract the weevil.",
               "- Record every check in the system so you can follow how the condition changes."],
        "ur": ["متاثرہ کھجور کے ساتھ والی کھجوروں کو بچانے کے لیے عمومی رہنمائی:",
               "- متاثرہ کھجور کی اطلاع مقامی زرعی محکمے کو دیں، اور ان کی ہدایت سے پہلے اسے نہ علاج کریں نہ کاٹیں۔",
               "- متاثرہ کھجور کی قلمیں، پتے یا کچرا کسی اور جگہ منتقل نہ کریں۔",
               "- ساتھ والی کھجوروں کا معمول سے زیادہ معائنہ کریں، خاص طور پر تنے کی جڑ اور چوٹی۔",
               "- جہاں تک ممکن ہو ساتھ والی کھجوروں کو زخمی نہ کریں (کٹائی یا قلمیں الگ کرنا)، کیونکہ تازہ زخم سونڈی کو کھینچتے ہیں۔",
               "- ہر جانچ سسٹم میں درج کریں تاکہ حالت کی تبدیلی دیکھ سکیں۔"],
    },
    "prune": {
        "ar": ["إرشاد عام عند تقليم النخيل:",
               "- لا تقلّم أو تجرح النخلة إلا للضرورة، لأن الجروح الطازجة وقواعد السعف المقطوعة تجذب إناث السوس لوضع البيض.",
               "- استخدم أدوات نظيفة ومعقّمة، ويفضَّل أن يكون العمل في الأجواء الباردة.",
               "- عالج مكان القطع بالمادة التي توصي بها جهة الزراعة المحلية.",
               "- لا تترك مخلفات التقليم في المزرعة.",
               "- افحص النخلة وما حولها بعد التقليم بأيام."],
        "en": ["General guidance when pruning palms:",
               "- Do not prune or wound a palm unless you have to, because fresh wounds and cut frond bases attract egg-laying female weevils.",
               "- Use clean, disinfected tools, and prefer cool weather for the work.",
               "- Treat the cut with the product your local agriculture authority recommends.",
               "- Do not leave pruning waste lying on the farm.",
               "- Inspect the palm and the ones around it a few days after pruning."],
        "ur": ["کھجور کی کٹائی کے وقت عمومی رہنمائی:",
               "- جب تک ضروری نہ ہو کھجور کو نہ کاٹیں نہ زخمی کریں، کیونکہ تازہ زخم اور کٹے ہوئے پتوں کی جڑیں انڈے دینے والی مادہ سونڈی کو کھینچتی ہیں۔",
               "- صاف اور جراثیم سے پاک اوزار استعمال کریں، اور بہتر ہے کہ کام ٹھنڈے موسم میں ہو۔",
               "- کٹی ہوئی جگہ پر وہی دوا لگائیں جو مقامی زرعی محکمہ تجویز کرے۔",
               "- کٹائی کا کچرا کھیت میں نہ چھوڑیں۔",
               "- کٹائی کے چند دن بعد اس کھجور اور آس پاس کی کھجوروں کا معائنہ کریں۔"],
    },
}

_ID = re.compile(r"palm[\s\-_]*([a-z])[\s\-_]*(\d{1,2})", re.I)
_DUE = re.compile(r"re-?check|due for|overdue|متأخر|متاخر|إعادة الفحص|اعادة الفحص|دوبارہ|تاخیر", re.I)
_ATT = re.compile(r"attention|needs?\b|urgent|عناية|انتباه|تحتاج|توجہ|توجه", re.I)
_PROTECT = re.compile(r"protect|neighbou?r|next to|المجاور|مجاور|أحمي|احمي|حماية|بچاؤ|بچا", re.I)
_PRUNE = re.compile(r"prun|تقليم|كٹائی|کٹائی", re.I)


def _days(iso):
    try:
        return max(0, (datetime.utcnow() - datetime.fromisoformat(iso)).days)
    except Exception:
        return 0


def ready_answer(message, language, tools):
    """tools: demo_agent.TOOL_EXECUTOR. Returns {"answer": str, "trace": [{"tool", "args"}]}."""
    language = language if language in TXT else "en"
    t, st = TXT[language], STATUS[language]
    msg = message or ""
    m = _ID.search(msg)
    trace, lines = [], []

    def status(p):
        return st.get(p["latest_status"], p["latest_status"])

    if not m and _PRUNE.search(msg):
        lines = GENERAL["prune"][language]
    elif not m and _PROTECT.search(msg):
        lines = GENERAL["protect"][language]
    elif not m and _DUE.search(msg):
        trace.append({"tool": "palms_due_for_recheck", "args": {}})
        due = tools["palms_due_for_recheck"]()
        lines = [t["due"]] + [t["dueline"](p["palm_id"], _ago(p["days_since_last_check"], language)) for p in due] if due else [t["none_due"]]
    elif m:
        pid = f"Palm-{m.group(1).upper()}{int(m.group(2)):02d}"
        trace.append({"tool": "get_palm_details", "args": {"palm_id": pid}})
        d = tools["get_palm_details"](pid)
        if "error" in d:
            lines = [t["notfound"]]
        else:
            lines = [t["det"](d["palm_id"], d["location"], d["age_years"])]
            if d.get("notes"):
                lines.append(t["notes"] + d["notes"])
            lines.append(t["hist"])
            for h in d["check_history"][-3:]:
                lines.append(t["histline"](st.get(h["class"], h["class"]), _ago(_days(h["date"]), language)))
    elif _ATT.search(msg):
        trace.append({"tool": "palms_needing_attention", "args": {}})
        att = tools["palms_needing_attention"]()
        lines = [t["att"]] + [t["line"](p["palm_id"], status(p), _ago(_days(p["latest_check"]), language)) for p in att] if att else [t["none_att"]]
    else:
        trace.append({"tool": "list_palms", "args": {}})
        lines = [t["other"], t["all"]] + [t["line"](p["palm_id"], status(p), _ago(_days(p["latest_check"]), language)) for p in tools["list_palms"]()]
    return {"answer": NOTE[language] + "\n" + "\n".join(lines), "trace": trace}
