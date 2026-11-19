"""
Treatment report for the public try page: short headings that open into the details, and a one-month plan, week by week,
with the actions day by day.

The AI assistant of NakhlaGuard (named "Haris" / "حارس" on the site; it is Gemini underneath) writes the content as JSON.
The headings are fixed here, so the page always looks the same. If the AI key is missing, the AI is busy, or its answer is
not usable, a ready-made report (the same structure, written by us) is returned and labelled as such, so the demo never
shows an empty or broken report.

What the report holds (the same in the AI version and in the ready-made version):
  meaning   what this result means, and that it must be confirmed in the field
  today     what to do today
  weeks     four weeks; each has a focus, a goal, the actions day by day, and what to check at the end of the week
  cost      why acting early matters (in words, no invented numbers)
  watch     signs that mean "call a specialist now"
Each week also carries a ready question for the assistant ("ask"), so the page can open the assistant with it.

General guidance only: no product names, no doses, no promise of a cure, nothing that replaces the field inspection by a specialist.
"""
import asyncio
import json
import re

AI_REPORT_TIMEOUT_S = 38   # the page already shows the ready-made version while the AI writes, so it can wait; after this the labelled ready-made report stays

STAGES = ("healthy", "early", "moderate", "advanced")

TITLES = {
    "ar": {"meaning": "ماذا تعني هذه النتيجة", "today": "ما أفعله اليوم", "plan": "خطة الشهر، أسبوعًا بأسبوع", "w1": "الأسبوع الأول", "w2": "الأسبوع الثاني",
           "w3": "الأسبوع الثالث", "w4": "الأسبوع الرابع", "cost": "لماذا لا نؤجّل", "watch": "متى أستدعي مختصًا",
           "check": "نتحقق في نهاية الأسبوع", "goal": "هدف الأسبوع", "ask_btn": "اسأل «حارس» عن هذا الأسبوع", "ask_all": "اسأل «حارس» عن خطتك",
           "stage": {"healthy": "سليمة", "early": "إصابة مبكرة", "moderate": "إصابة متوسطة", "advanced": "إصابة متقدمة"},
           "note": "إرشادات عامة وليست وصفة علاج. الحكم النهائي للمختص بعد الفحص الميداني."},
    "en": {"meaning": "What this result means", "today": "What to do today", "plan": "The month plan, week by week", "w1": "Week 1", "w2": "Week 2", "w3": "Week 3", "w4": "Week 4",
           "cost": "Why not to wait", "watch": "When to call a specialist",
           "check": "Check at the end of the week", "goal": "Goal of the week", "ask_btn": "Ask Haris about this week", "ask_all": "Ask Haris about your plan",
           "stage": {"healthy": "healthy", "early": "early infestation", "moderate": "moderate infestation", "advanced": "advanced infestation"},
           "note": "General guidance, not a treatment prescription. The specialist decides after the field inspection."},
    "ur": {"meaning": "اس نتیجے کا مطلب", "today": "آج کیا کرنا ہے", "plan": "مہینے کا منصوبہ، ہفتہ بہ ہفتہ", "w1": "پہلا ہفتہ", "w2": "دوسرا ہفتہ", "w3": "تیسرا ہفتہ", "w4": "چوتھا ہفتہ",
           "cost": "تاخیر کیوں نہ کریں", "watch": "ماہر کو کب بلانا ہے",
           "check": "ہفتے کے آخر میں جانچیں", "goal": "ہفتے کا ہدف", "ask_btn": "اس ہفتے کے بارے میں «حارس» سے پوچھیں", "ask_all": "اپنے منصوبے کے بارے میں «حارس» سے پوچھیں",
           "stage": {"healthy": "صحت مند", "early": "ابتدائی حملہ", "moderate": "درمیانہ حملہ", "advanced": "شدید حملہ"},
           "note": "عمومی رہنمائی ہے، علاج کا نسخہ نہیں۔ میدانی معائنے کے بعد فیصلہ ماہر کرتا ہے۔"},
}

# ---------------------------------------------------------------- ready-made reports (3 groups x 3 languages)
# each: headline per stage, meaning, today, four weeks {focus, goal, tasks (day by day), check}, cost, watch
_T = {
    "healthy": {
        "ar": {
            "headline": {"healthy": "لا مؤشرات إصابة في هذا الصوت. خطتك هذا الشهر وقاية ومتابعة منتظمة."},
            "meaning": ["لم يسمع الجهاز نبضات قصيرة متكررة تشبه نشاط اليرقات في هذا التسجيل.",
                        "هذا لا يعني أن النخلة محصّنة: تسجيل واحد لا يغني عن النظر المنتظم والفحص الميداني.",
                        "النموذج تدرّب على تسجيلات محاكاة ولم يُختبر على نخيل حقيقي، فاعتبر النتيجة مؤشرًا أوليًا فقط."],
            "today": ["احفظ هذا الفحص كقراءة أساس للنخلة: التاريخ والمكان والتسجيل.", "تفقّد الجذع وقاعدة السعف بحثًا عن ثقوب أو سائل بني أو نشارة أو رائحة تخمّر.",
                      "لا تقلّم النخلة ولا تجرحها إلا لضرورة، فالجروح الطازجة تجذب السوسة."],
            "weeks": [
                {"focus": "خط الأساس", "goal": "أن تعرف حالة نخيلك اليوم لتقارن بها لاحقًا.",
                 "tasks": ["اليوم 1: سجّل النخلة في موضع محدد على الجذع، وصوّر الموضع أو علّمه حتى تعيد التسجيل في المكان نفسه.",
                           "اليوم 2: افحص بالنظر قاعدة الجذع وأسفل السعف والفسائل بحثًا عن ثقوب أو إفرازات أو نشارة.",
                           "اليومان 3 و4: سجّل نخلتين مجاورتين بالطريقة نفسها لتكون عندك مقارنة.",
                           "اليوم 7: دوّن ما رأيته في ورقة أو ملف واحد: رقم النخلة وتاريخ الفحص وعدد النبضات."],
                 "check": "هل عندك قراءة مدوّنة لكل نخلة فحصتها، وهل تجد الموضع نفسه بسهولة؟"},
                {"focus": "نظافة المزرعة", "goal": "إزالة ما يجذب السوسة ويخبّئها حول النخيل.",
                 "tasks": ["اليومان 1 و2: أزل السعف اليابس والمخلفات المتراكمة حول النخيل وانقلها بعيدًا عن المزرعة.",
                           "اليوم 3: لا تترك فسائل مقطوعة أو سعفًا مقلّمًا ملقى قرب النخيل.",
                           "اليوم 4: إن كان التقليم ضروريًا فاسأل المختص عن وقته المناسب وكيف تعالج مكان القطع.",
                           "اليومان 5 و6: افحص مصادر الماء ومواضع الرطوبة عند الجذوع، فالرطوبة تجذب الحشرة.",
                           "اليوم 7: صوّر حال الأرض حول النخيل لتقارن بها."],
                 "check": "هل خلت الأرض حول النخيل من المخلفات والفسائل المقطوعة؟"},
                {"focus": "المصائد والإرشاد", "goal": "ربط مزرعتك ببرنامج المكافحة المحلي والاستعداد المبكر.",
                 "tasks": ["اليوم 1: اتصل بمكتب الإرشاد الزراعي في منطقتك واسأل عن برنامج مكافحة سوسة النخيل المحلي.",
                           "اليوم 2: اسأل عن مصائد الفرمون: هل تتوفر لك، وكم عددها، وأين توضع، ومن يتابعها.",
                           "اليومان 3 و4: إن وُفّرت المصائد فعلّقها بحسب إرشاد المختص، وسجّل تاريخ التركيب.",
                           "اليوم 7: افحص المصائد وسجّل ما فيها، حتى لو كانت فارغة."],
                 "check": "هل تعرف الجهة التي تتصل بها فورًا إن ظهرت علامة، وهل المصائد (إن وُجدت) تعمل؟"},
                {"focus": "إعادة الفحص", "goal": "مقارنة الحال بعد شهر وتحديد موعد المتابعة التالية.",
                 "tasks": ["اليوم 1: أعد تسجيل كل نخلة في الموضع نفسه وقارن بقراءة الأسبوع الأول.",
                           "اليوم 2: افحص الجذوع والقلب بالنظر من جديد.",
                           "اليوم 3: إن تغيّر شيء (عدد النبضات أو علامة ظاهرة) فاطلب فحصًا ميدانيًا ولا تنتظر.",
                           "اليوم 7: حدّد موعد الفحص القادم بعد نحو 30 يومًا وضع له تذكيرًا."],
                 "check": "هل القراءات متقاربة مع الأسبوع الأول، وهل حُدد موعد الفحص القادم؟"}],
            "cost": ["الوقاية أرخص من العلاج: النخلة التي تُكتشف إصابتها مبكرًا يسهل إنقاذها، وبعد تقدم الإصابة قد لا تُنقذ.",
                     "كل أسبوع تأخير بعد ظهور علامة يتيح لليرقات الوصول إلى مناطق أعمق من الجذع."],
            "watch": ["ظهور ثقوب أو سائل بني أو نشارة أو رائحة تخمّر.", "ذبول القلب (السعف الأوسط) أو ميلانه.", "أي نتيجة غير سليمة في فحص لاحق."]},
        "en": {
            "headline": {"healthy": "No sign of infestation in this sound. Your plan this month is prevention and regular monitoring."},
            "meaning": ["The device did not hear short, repeated pulses like larval activity in this recording.",
                        "This does not make the palm immune: one recording does not replace regular looking and a field inspection.",
                        "The model was trained on simulated recordings and has not been tested on real palms, so treat this as a first indication only."],
            "today": ["Save this check as the palm's baseline: date, place and recording.", "Look at the trunk and the frond bases for holes, brown fluid, sawdust or a fermenting smell.",
                      "Do not prune or wound the palm unless you must: fresh wounds attract the weevil."],
            "weeks": [
                {"focus": "Baseline", "goal": "Know the state of your palms today so you can compare later.",
                 "tasks": ["Day 1: record the palm at one fixed spot on the trunk, and photograph or mark the spot so you can return to it.",
                           "Day 2: look over the trunk base, under the fronds and the offshoots for holes, oozing or sawdust.",
                           "Days 3-4: record two neighbouring palms the same way, to have a comparison.",
                           "Day 7: write down what you saw in one place: palm number, date and number of pulses."],
                 "check": "Do you have a written reading for every palm you checked, and can you find the same spot easily?"},
                {"focus": "Keep the farm clean", "goal": "Remove what attracts the weevil and hides it around the palms.",
                 "tasks": ["Days 1-2: remove dry fronds and debris piled around the palms and carry them away from the farm.",
                           "Day 3: do not leave cut offshoots or pruned fronds lying near the palms.",
                           "Day 4: if pruning is necessary, ask a specialist about the right time and how to treat the cut.",
                           "Days 5-6: check water sources and damp spots at the trunks; moisture attracts the insect.",
                           "Day 7: photograph the ground around the palms to compare later."],
                 "check": "Is the ground around the palms free of debris and cut offshoots?"},
                {"focus": "Traps and advice", "goal": "Connect your farm to the local control programme and be ready early.",
                 "tasks": ["Day 1: call your local agricultural extension office and ask about the local red palm weevil programme.",
                           "Day 2: ask about pheromone traps: whether you can get them, how many, where to place them and who checks them.",
                           "Days 3-4: if traps are provided, hang them as the specialist advises and write down the date.",
                           "Day 7: check the traps and write down what is in them, even if empty."],
                 "check": "Do you know who to call at once if a sign appears, and are the traps (if any) in place?"},
                {"focus": "Check again", "goal": "Compare after a month and set the next check.",
                 "tasks": ["Day 1: record every palm again at the same spot and compare with week 1.",
                           "Day 2: look at the trunks and the crown again.",
                           "Day 3: if anything changed (pulse count or a visible sign), ask for a field inspection and do not wait.",
                           "Day 7: set the next check for about 30 days later and put a reminder on it."],
                 "check": "Are the readings close to week 1, and is the next check set?"}],
            "cost": ["Prevention is cheaper than treatment: a palm caught early is easy to save, while an advanced one may be lost.",
                     "Each week of delay after a sign appears lets the larvae reach deeper parts of the trunk."],
            "watch": ["Holes, brown fluid, sawdust or a fermenting smell.", "A wilting or leaning crown (the middle fronds).", "Any result other than healthy in a later check."]},
        "ur": {
            "headline": {"healthy": "اس آواز میں حملے کی علامت نہیں۔ اس مہینے آپ کا منصوبہ بچاؤ اور باقاعدہ نگرانی ہے۔"},
            "meaning": ["اس ریکارڈنگ میں ڈیوائس نے سنڈی کی سرگرمی جیسی مختصر اور بار بار آنے والی پلس نہیں سنیں۔",
                        "اس کا مطلب یہ نہیں کہ کھجور محفوظ ہے: ایک ریکارڈنگ باقاعدہ دیکھ بھال اور میدانی معائنے کی جگہ نہیں لیتی۔",
                        "ماڈل سمولیشن ریکارڈنگز پر تربیت یافتہ ہے اور حقیقی کھجوروں پر آزمایا نہیں گیا، اس لیے اسے صرف ابتدائی اشارہ سمجھیں۔"],
            "today": ["اس جانچ کو کھجور کی بنیادی ریڈنگ کے طور پر محفوظ کریں: تاریخ، جگہ اور ریکارڈنگ۔", "تنے اور پتوں کی جڑوں میں سوراخ، بھورا مائع، برادہ یا خمیر کی بو دیکھیں۔",
                      "ضرورت کے بغیر کھجور کی کٹائی نہ کریں اور اسے زخمی نہ کریں: تازہ زخم سنڈی کو کھینچتے ہیں۔"],
            "weeks": [
                {"focus": "بنیادی ریڈنگ", "goal": "آج اپنی کھجوروں کی حالت جانیں تاکہ بعد میں موازنہ ہو سکے۔",
                 "tasks": ["دن 1: کھجور کو تنے پر ایک مقررہ جگہ ریکارڈ کریں اور اس جگہ کی تصویر لیں یا نشان لگائیں۔",
                           "دن 2: تنے کی جڑ، پتوں کے نیچے اور پودوں میں سوراخ، رسنے یا برادے کو دیکھیں۔",
                           "دن 3 اور 4: دو پاس کی کھجوریں اسی طرح ریکارڈ کریں تاکہ موازنہ ہو۔",
                           "دن 7: جو دیکھا ایک جگہ لکھ لیں: کھجور کا نمبر، تاریخ اور پلس کی تعداد۔"],
                 "check": "کیا ہر جانچی گئی کھجور کی تحریری ریڈنگ آپ کے پاس ہے، اور کیا وہی جگہ آسانی سے مل جاتی ہے؟"},
                {"focus": "باغ کی صفائی", "goal": "وہ چیزیں ہٹانا جو کھجوروں کے گرد سنڈی کو کھینچتی اور چھپاتی ہیں۔",
                 "tasks": ["دن 1 اور 2: کھجوروں کے گرد سوکھے پتے اور کچرا ہٹا کر باغ سے دور لے جائیں۔",
                           "دن 3: کٹے ہوئے پودے یا کٹے پتے کھجوروں کے پاس نہ پڑے رہنے دیں۔",
                           "دن 4: کٹائی ضروری ہو تو ماہر سے مناسب وقت اور کٹی جگہ کے علاج کا پوچھیں۔",
                           "دن 5 اور 6: پانی کے ذرائع اور تنوں کے پاس نمی کی جگہیں دیکھیں؛ نمی کیڑے کو کھینچتی ہے۔",
                           "دن 7: کھجوروں کے گرد زمین کی تصویر لیں تاکہ بعد میں موازنہ ہو۔"],
                 "check": "کیا کھجوروں کے گرد زمین کچرے اور کٹے ہوئے پودوں سے خالی ہے؟"},
                {"focus": "پھندے اور مشورہ", "goal": "اپنے باغ کو مقامی کنٹرول پروگرام سے جوڑنا اور جلد تیار رہنا۔",
                 "tasks": ["دن 1: اپنے علاقے کے زرعی توسیعی دفتر کو فون کر کے مقامی ریڈ پام ویول پروگرام کا پوچھیں۔",
                           "دن 2: فیرومون پھندوں کا پوچھیں: ملیں گے یا نہیں، کتنے، کہاں لگیں اور کون دیکھے گا۔",
                           "دن 3 اور 4: پھندے ملیں تو ماہر کے مشورے کے مطابق لگائیں اور تاریخ لکھ لیں۔",
                           "دن 7: پھندے دیکھیں اور جو کچھ ان میں ہے لکھیں، خالی ہوں تب بھی۔"],
                 "check": "کیا آپ کو معلوم ہے کہ علامت ظاہر ہو تو فوراً کسے فون کرنا ہے، اور کیا پھندے (اگر ہیں) لگ چکے ہیں؟"},
                {"focus": "دوبارہ جانچ", "goal": "ایک مہینے بعد موازنہ کرنا اور اگلی جانچ طے کرنا۔",
                 "tasks": ["دن 1: ہر کھجور اسی جگہ دوبارہ ریکارڈ کریں اور پہلے ہفتے سے موازنہ کریں۔",
                           "دن 2: تنوں اور چوٹی کو دوبارہ دیکھیں۔",
                           "دن 3: کچھ بدلا ہو (پلس کی تعداد یا ظاہری علامت) تو میدانی معائنہ مانگیں اور انتظار نہ کریں۔",
                           "دن 7: اگلی جانچ تقریباً 30 دن بعد طے کریں اور یاد دہانی لگا لیں۔"],
                 "check": "کیا ریڈنگز پہلے ہفتے کے قریب ہیں، اور کیا اگلی جانچ طے ہو گئی؟"}],
            "cost": ["بچاؤ علاج سے سستا ہے: جلد پکڑی گئی کھجور بچانا آسان ہے، جبکہ بڑھا ہوا حملہ کھجور کو ضائع کر سکتا ہے۔",
                     "علامت ظاہر ہونے کے بعد ہر ہفتے کی تاخیر سے سنڈیاں تنے کے گہرے حصوں تک پہنچ جاتی ہیں۔"],
            "watch": ["سوراخ، بھورا مائع، برادہ یا خمیر کی بو۔", "چوٹی (درمیانے پتوں) کا مرجھانا یا جھکنا۔", "بعد کی جانچ میں صحت مند کے علاوہ کوئی نتیجہ۔"]},
    },
    "infested": {
        "ar": {
            "headline": {"early": "مؤشرات إصابة مبكرة. الأسابيع الأولى هي الأهم: أكّد الإصابة ميدانيًا وابدأ العلاج بسرعة.",
                         "moderate": "إصابة متوسطة محتملة. تحتاج تأكيدًا ميدانيًا وعلاجًا بإشراف مختص خلال أيام."},
            "meaning": ["سمع الجهاز نبضات قصيرة متكررة تشبه نشاط اليرقات داخل الجذع، وصنّفها النموذج إصابة محتملة.",
                        "هذه مؤشرات وليست تشخيصًا: النموذج تدرّب على تسجيلات محاكاة ولم يُختبر على نخيل حقيقي، فلا بد من التأكيد ميدانيًا.",
                        "كلما أُكدت الإصابة وعولجت أبكر، زادت فرصة إنقاذ النخلة وحماية جيرانها."],
            "today": ["علّم النخلة وأبعد عنها كل نشاط غير ضروري.", "اتصل بالمرشد الزراعي أو بوزارة البيئة والمياه والزراعة واطلب فحصًا ميدانيًا.",
                      "لا تقلّم ولا تنقل أي جزء من النخلة أو فسائلها خارج المزرعة.", "افحص النخيل المجاورة بالنظر اليوم وسجّلها بالجهاز."],
            "weeks": [
                {"focus": "التأكيد والعزل", "goal": "تأكيد الإصابة ميدانيًا ومنع انتقالها إلى النخيل المجاورة.",
                 "tasks": ["اليوم 1: اتصل بالمختص (المرشد الزراعي أو الوزارة) واطلب موعد فحص، وعلّم النخلة المشتبه بها.",
                           "اليوم 1: امنع التقليم ونقل السعف والفسائل والمخلفات من هذه النخلة إلى خارج المزرعة.",
                           "اليومان 2 و3: افحص كل نخلة في محيط النخلة بالنظر (ثقوب، سائل بني، نشارة، ذبول القلب) وسجّلها بالجهاز مرتين.",
                           "اليوم 4: دوّن ما رأيته وصوّر العلامات الظاهرة لتعرضها على المختص.",
                           "اليومان 5 إلى 7: استقبل المختص، ودعه يؤكد الإصابة ويحدد مداها ويقرر مسار العلاج."],
                 "check": "هل أُكدت الإصابة أو نُفيت من مختص، وهل عرفت أي نخلة أخرى تحتاج فحصًا؟"},
                {"focus": "العلاج", "goal": "تنفيذ العلاج الذي يقرره المختص كما هو، وبأمان.",
                 "tasks": ["اليوم 1: اطلب من المختص كتابة خطة العلاج: ماذا يُستعمل ومتى ومن ينفذه.",
                           "اليومان 1 و2: ينفذ المختص (أو تنفذ أنت بتوجيهه) تنظيف الأنسجة المصابة وعلاج الجروح بالطريقة الصحيحة.",
                           "اليوم 3: لا تستعمل أي مادة بلا إشراف ولا تخلط مواد من عندك.",
                           "اليوم 4: ابعد الناس والأطفال عن النخلة إن بدا القلب ضعيفًا أو الجذع مائلًا.",
                           "اليومان 5 و6: ثبّت مصائد الفرمون بإرشاد المختص وسجّل تاريخ التركيب.",
                           "اليوم 7: سجّل النخلة مرة واحدة ودوّن أي تغيّر ظهر عليها."],
                 "check": "هل نُفّذ العلاج كما كُتب في الخطة، وهل سجّلت تاريخه وما استُعمل فيه؟"},
                {"focus": "المتابعة", "goal": "معرفة هل بدأ النشاط يهدأ، وحماية النخيل المجاورة.",
                 "tasks": ["اليوم 1: أعد تسجيل النخلة ثلاث مرات في الموضع نفسه والوقت نفسه من اليوم، وقارن عدد النبضات بالأسبوع الأول.",
                           "اليومان 2 و3: افحص المصائد وسجّل ما فيها، وافحص النخيل المجاورة مرة أخرى.",
                           "اليوم 4: أبلغ المختص بالنتائج، واسأله هل يلزم زيارة علاج ثانية.",
                           "اليومان 5 و6: نظّف الأرض حول النخيل من السعف اليابس والمخلفات.",
                           "اليوم 7: دوّن ملخص الأسبوع بالأرقام."],
                 "check": "هل انخفض عدد النبضات عن الأسبوع الأول، وهل ظهرت علامة على نخلة مجاورة؟"},
                {"focus": "التقييم", "goal": "الحكم على نجاح العلاج وتحديد الخطوة التالية مع المختص.",
                 "tasks": ["اليومان 1 و2: أعد تسجيل كل النخيل القريبة وقارن قراءات الأسابيع الأول والثالث والرابع.",
                           "اليوم 3: أرسل الملخص للمختص ليقرر: متابعة، أو علاج ثان، أو إزالة النخلة إن لزم.",
                           "اليوم 4: إن انخفض النشاط فاستمر بالمتابعة كل أسبوعين.",
                           "اليوم 5: إن لم ينخفض فاطلب تقييمًا عاجلًا ولا تنتظر الشهر القادم.",
                           "اليومان 6 و7: حدّث سجل المزرعة بكل ما فُعل وتواريخه."],
                 "check": "هل اتُّخذ قرار واضح (متابعة أو علاج ثان أو إزالة) مع المختص ومدوّن عندك؟"}],
            "cost": ["علاج الإصابة المبكرة أيسر وأقل كلفة غالبًا من علاج نخلة متقدمة الإصابة أو خسارتها.",
                     "اليرقات تتغذى داخل الجذع بعيدًا عن النظر، فكل أسبوع تأخير قد يوصلها إلى القلب، وهو أخطر موضع.",
                     "تأخير الإبلاغ قد ينقل السوسة إلى النخيل المجاورة فيتضاعف الضرر."],
            "watch": ["زيادة النبضات في فحص لاحق.", "سائل بني أو نشارة أو ثقوب أو ذبول في القلب.", "ظهور أعراض على نخلة مجاورة.", "تأخر وصول المختص أكثر من بضعة أيام: أعد الاتصال."]},
        "en": {
            "headline": {"early": "Early signs of infestation. The first weeks matter most: confirm it in the field and start treatment fast.",
                         "moderate": "A moderate infestation is possible. It needs field confirmation and treatment under a specialist within days."},
            "meaning": ["The device heard short, repeated pulses like larval activity inside the trunk, and the model rated it a possible infestation.",
                        "These are signs, not a diagnosis: the model was trained on simulated recordings and has not been tested on real palms, so it must be confirmed in the field.",
                        "The earlier an infestation is confirmed and treated, the better the chance of saving the palm and protecting its neighbours."],
            "today": ["Mark the palm and keep unnecessary activity away from it.", "Call the agricultural extension service or the Ministry of Environment, Water and Agriculture and ask for a field inspection.",
                      "Do not prune, and do not move any part of the palm or its offshoots off the farm.", "Look over the neighbouring palms today and record them with the device."],
            "weeks": [
                {"focus": "Confirm and contain", "goal": "Confirm the infestation in the field and stop it reaching the neighbouring palms.",
                 "tasks": ["Day 1: call the specialist (extension service or the Ministry), ask for an inspection date, and mark the suspected palm.",
                           "Day 1: forbid pruning and moving fronds, offshoots or debris from this palm off the farm.",
                           "Days 2-3: look over every palm around it (holes, brown fluid, sawdust, wilting crown) and record each one twice with the device.",
                           "Day 4: write down what you saw and photograph the visible signs to show the specialist.",
                           "Days 5-7: receive the specialist, who confirms the infestation, sets its extent and decides the treatment path."],
                 "check": "Has a specialist confirmed or ruled out the infestation, and do you know which other palms need checking?"},
                {"focus": "Treatment", "goal": "Carry out the treatment the specialist decides, as given, and safely.",
                 "tasks": ["Day 1: ask the specialist to write down the treatment plan: what is used, when and who does it.",
                           "Days 1-2: the specialist (or you, under their direction) cleans the infested tissue and treats the wounds the right way.",
                           "Day 3: do not use any product without supervision and do not mix products yourself.",
                           "Day 4: keep people and children away from the palm if the crown looks weak or the trunk leans.",
                           "Days 5-6: set up pheromone traps as the specialist advises and write down the date.",
                           "Day 7: record the palm once and note any change you see."],
                 "check": "Was the treatment done as written in the plan, and did you note its date and what was used?"},
                {"focus": "Follow-up", "goal": "Learn whether the activity is calming, and protect the neighbouring palms.",
                 "tasks": ["Day 1: record the palm three times at the same spot and the same time of day, and compare the pulse count with week 1.",
                           "Days 2-3: check the traps and write down what is in them, and check the neighbouring palms again.",
                           "Day 4: tell the specialist the results and ask whether a second treatment visit is needed.",
                           "Days 5-6: clean the ground around the palms of dry fronds and debris.",
                           "Day 7: write the week's summary with numbers."],
                 "check": "Is the pulse count lower than in week 1, and has a sign appeared on a neighbouring palm?"},
                {"focus": "Review", "goal": "Judge whether the treatment worked and decide the next step with the specialist.",
                 "tasks": ["Days 1-2: record all the nearby palms again and compare the readings of weeks 1, 3 and 4.",
                           "Day 3: send the summary to the specialist to decide: keep monitoring, treat again, or remove the palm if needed.",
                           "Day 4: if activity dropped, keep monitoring every two weeks.",
                           "Day 5: if it did not drop, ask for an urgent assessment and do not wait for next month.",
                           "Days 6-7: update the farm record with everything done and the dates."],
                 "check": "Is there a clear decision (keep monitoring, treat again or remove) made with the specialist and written down?"}],
            "cost": ["Treating an early infestation is usually easier and cheaper than treating an advanced one or losing the palm.",
                     "The larvae feed inside the trunk out of sight, so each week of delay can bring them closer to the crown, the most dangerous place.",
                     "A late report can let the weevil reach neighbouring palms, and then the damage multiplies."],
            "watch": ["More pulses in a later check.", "Brown fluid, sawdust, holes or a wilting crown.", "Signs on a neighbouring palm.", "The specialist has not come after a few days: call again."]},
        "ur": {
            "headline": {"early": "حملے کی ابتدائی علامات۔ ابتدائی ہفتے سب سے اہم ہیں: میدان میں تصدیق کریں اور جلد علاج شروع کریں۔",
                         "moderate": "درمیانے حملے کا امکان۔ چند دنوں میں میدانی تصدیق اور ماہر کی نگرانی میں علاج چاہیے۔"},
            "meaning": ["ڈیوائس نے تنے کے اندر سنڈی کی سرگرمی جیسی مختصر اور بار بار آنے والی پلس سنیں، اور ماڈل نے اسے ممکنہ حملہ قرار دیا۔",
                        "یہ علامات ہیں، تشخیص نہیں: ماڈل سمولیشن ریکارڈنگز پر تربیت یافتہ ہے اور حقیقی کھجوروں پر آزمایا نہیں گیا، اس لیے میدان میں تصدیق ضروری ہے۔",
                        "حملے کی تصدیق اور علاج جتنا جلد ہو، کھجور بچانے اور اس کے پڑوسیوں کو محفوظ رکھنے کا امکان اتنا بہتر ہے۔"],
            "today": ["کھجور پر نشان لگائیں اور اس کے پاس غیر ضروری سرگرمی روکیں۔", "زرعی توسیعی سروس یا وزارتِ ماحولیات، پانی و زراعت کو فون کر کے میدانی معائنہ مانگیں۔",
                      "کٹائی نہ کریں اور کھجور یا اس کے پودوں کا کوئی حصہ باغ سے باہر نہ لے جائیں۔", "آج پاس کی کھجوروں کو غور سے دیکھیں اور ڈیوائس سے ریکارڈ کریں۔"],
            "weeks": [
                {"focus": "تصدیق اور علیحدگی", "goal": "میدان میں حملے کی تصدیق کرنا اور اسے پاس کی کھجوروں تک پہنچنے سے روکنا۔",
                 "tasks": ["دن 1: ماہر (توسیعی سروس یا وزارت) کو فون کر کے معائنے کی تاریخ مانگیں اور مشتبہ کھجور پر نشان لگائیں۔",
                           "دن 1: اس کھجور کے پتے، پودے اور کچرا باغ سے باہر لے جانا اور کٹائی بند کر دیں۔",
                           "دن 2 اور 3: اس کے گرد ہر کھجور کو دیکھیں (سوراخ، بھورا مائع، برادہ، چوٹی کا مرجھانا) اور ہر ایک کو ڈیوائس سے دو بار ریکارڈ کریں۔",
                           "دن 4: جو دیکھا لکھ لیں اور ظاہری علامات کی تصویر لیں تاکہ ماہر کو دکھا سکیں۔",
                           "دن 5 سے 7: ماہر کو آنے دیں، وہ حملے کی تصدیق کرے گا، اس کی حد بتائے گا اور علاج کا راستہ طے کرے گا۔"],
                 "check": "کیا ماہر نے حملے کی تصدیق یا تردید کی، اور کیا آپ کو معلوم ہے کہ کن اور کھجوروں کی جانچ باقی ہے؟"},
                {"focus": "علاج", "goal": "ماہر کا طے کردہ علاج جیسا ہے ویسا اور محفوظ طریقے سے کرنا۔",
                 "tasks": ["دن 1: ماہر سے علاج کا منصوبہ لکھوائیں: کیا استعمال ہوگا، کب اور کون کرے گا۔",
                           "دن 1 اور 2: ماہر (یا اس کی ہدایت پر آپ) متاثرہ حصہ صاف کرتا ہے اور زخموں کا صحیح علاج کرتا ہے۔",
                           "دن 3: کوئی چیز بغیر نگرانی استعمال نہ کریں اور خود چیزیں نہ ملائیں۔",
                           "دن 4: چوٹی کمزور لگے یا تنا جھکا ہو تو لوگوں اور بچوں کو کھجور سے دور رکھیں۔",
                           "دن 5 اور 6: ماہر کے مشورے سے فیرومون پھندے لگائیں اور تاریخ لکھ لیں۔",
                           "دن 7: کھجور کو ایک بار ریکارڈ کریں اور جو تبدیلی ہو لکھیں۔"],
                 "check": "کیا علاج منصوبے کے مطابق ہوا، اور کیا اس کی تاریخ اور استعمال شدہ چیز آپ نے لکھی؟"},
                {"focus": "فالو اپ", "goal": "معلوم کرنا کہ سرگرمی کم ہو رہی ہے یا نہیں، اور پاس کی کھجوروں کی حفاظت کرنا۔",
                 "tasks": ["دن 1: کھجور کو اسی جگہ اور دن کے اسی وقت تین بار ریکارڈ کریں اور پلس کی تعداد پہلے ہفتے سے ملائیں۔",
                           "دن 2 اور 3: پھندے دیکھیں اور جو ان میں ہو لکھیں، اور پاس کی کھجوروں کو دوبارہ جانچیں۔",
                           "دن 4: ماہر کو نتائج بتائیں اور پوچھیں کہ کیا دوسرا علاج درکار ہے۔",
                           "دن 5 اور 6: کھجوروں کے گرد زمین سے سوکھے پتے اور کچرا صاف کریں۔",
                           "دن 7: ہفتے کا خلاصہ اعداد کے ساتھ لکھیں۔"],
                 "check": "کیا پلس کی تعداد پہلے ہفتے سے کم ہے، اور کیا پاس کی کسی کھجور پر علامت ظاہر ہوئی؟"},
                {"focus": "جائزہ", "goal": "علاج کی کامیابی کا اندازہ لگانا اور ماہر کے ساتھ اگلا قدم طے کرنا۔",
                 "tasks": ["دن 1 اور 2: قریب کی تمام کھجوریں دوبارہ ریکارڈ کریں اور پہلے، تیسرے اور چوتھے ہفتے کی ریڈنگز ملائیں۔",
                           "دن 3: خلاصہ ماہر کو بھیجیں تاکہ وہ فیصلہ کرے: نگرانی جاری، دوبارہ علاج، یا ضرورت ہو تو کھجور ہٹانا۔",
                           "دن 4: سرگرمی کم ہوئی ہو تو ہر دو ہفتے بعد نگرانی جاری رکھیں۔",
                           "دن 5: کم نہ ہوئی ہو تو فوری جائزہ مانگیں اور اگلے مہینے کا انتظار نہ کریں۔",
                           "دن 6 اور 7: باغ کے ریکارڈ میں جو کیا اور کب کیا، سب لکھ لیں۔"],
                 "check": "کیا ماہر کے ساتھ واضح فیصلہ (نگرانی، دوبارہ علاج یا ہٹانا) ہوا اور لکھا گیا؟"}],
            "cost": ["ابتدائی حملے کا علاج عموماً بڑھے ہوئے حملے کے علاج یا کھجور کے نقصان سے آسان اور سستا ہوتا ہے۔",
                     "سنڈیاں تنے کے اندر نظر سے دور کھاتی ہیں، اس لیے ہر ہفتے کی تاخیر انہیں سب سے خطرناک جگہ، یعنی چوٹی، کے قریب پہنچا سکتی ہے۔",
                     "دیر سے اطلاع سے کیڑا پاس کی کھجوروں تک پہنچ سکتا ہے اور نقصان کئی گنا بڑھ جاتا ہے۔"],
            "watch": ["بعد کی جانچ میں پلس زیادہ ہوں۔", "بھورا مائع، برادہ، سوراخ یا چوٹی کا مرجھانا۔", "پاس کی کھجور پر علامات۔", "ماہر چند دن بعد بھی نہ آئے تو دوبارہ فون کریں۔"]},
    },
    "advanced": {
        "ar": {
            "headline": {"advanced": "إصابة متقدمة محتملة. تحتاج تدخلًا عاجلًا: أكّد ميدانيًا خلال أيام وقرّر مع المختص بين العلاج والإزالة."},
            "meaning": ["سمع الجهاز نبضات كثيفة ومتكررة تشبه نشاط يرقات كثيرة داخل الجذع، وصنّفها النموذج إصابة متقدمة محتملة.",
                        "هذه مؤشرات وليست تشخيصًا: النموذج تدرّب على تسجيلات محاكاة ولم يُختبر على نخيل حقيقي، فلا بد من التأكيد الميداني العاجل.",
                        "في الإصابة المتقدمة قد يكون القرار بين العلاج والإزالة، ويقرره المختص بعد الفحص."],
            "today": ["اتصل بالمرشد الزراعي أو بوزارة البيئة والمياه والزراعة اليوم واطلب فحصًا عاجلًا.", "اعزل النخلة وامنع نقل أي جزء منها أو من فسائلها.",
                      "أبعد الناس والأطفال عن النخلة إن بدا القلب ضعيفًا أو الجذع مائلًا.", "لا تعالجها بنفسك بمواد غير مرخّصة."],
            "weeks": [
                {"focus": "التدخل العاجل", "goal": "الوصول إلى قرار المختص بسرعة وعزل النخلة حتى لا تنتشر السوسة.",
                 "tasks": ["اليوم 1: اتصل بالمختص واطلب فحصًا عاجلًا، وكرر الاتصال في اليوم التالي إن لم يأتِ أحد.",
                           "اليوم 1: اعزل النخلة، وامنع الزوار والآليات، ولا تنقل منها سعفًا أو فسائل أو مخلفات.",
                           "اليومان 1 و2: افحص كل النخيل القريبة بالنظر (ثقوب، سائل بني، نشارة، ذبول القلب) وسجّلها بالجهاز.",
                           "اليوم 3: دوّن مكان النخلة وصوّرها وجهّز لها موضعًا يصل إليه المختص والآليات.",
                           "اليومان 4 إلى 7: استقبل المختص ودعه يقرر: علاج أم إزالة، وما يحتاجه كل قرار منك."],
                 "check": "هل حضر المختص وقرر كتابةً بين العلاج والإزالة، وهل فُحصت كل النخيل القريبة؟"},
                {"focus": "العلاج أو الإزالة", "goal": "تنفيذ قرار المختص بأمان وبطريقة تمنع انتشار الحشرة.",
                 "tasks": ["اليوم 1: ثبّت مع المختص موعد التنفيذ ومن سينفذ، وأخلِ محيط النخلة من الناس.",
                           "الأيام 2 إلى 4: إن أمكن إنقاذها فاتبع خطة العلاج كما يحددها المختص، ولا تستعمل مواد من عندك.",
                           "الأيام 2 إلى 4: وإن تقررت الإزالة فتُنفَّذ بإشراف المختص وتُتلف النخلة بالطريقة التي يحددها حتى لا تنتشر الحشرة.",
                           "اليوم 5: لا تترك جذعًا أو سعفًا أو مخلفات من النخلة في المزرعة أو تنقلها إلى مكان آخر.",
                           "اليومان 6 و7: دوّن ما نُفّذ وتاريخه واحتفظ بتوصيات المختص."],
                 "check": "هل نُفّذ قرار المختص كاملًا، وأُتلفت المخلفات أو عولجت كما طُلب؟"},
                {"focus": "حماية النخيل المجاورة", "goal": "منع انتقال الإصابة إلى النخيل التي حولها.",
                 "tasks": ["اليومان 1 و2: افحص النخيل المجاورة بالجهاز وبالنظر، وسجّل كل نخلة مرتين.",
                           "اليوم 3: نفّذ المعالجة الوقائية التي يوصي بها المختص للنخيل القريبة.",
                           "اليومان 4 و5: ثبّت مصائد الفرمون بإرشاد المختص وسجّل مواضعها.",
                           "اليوم 6: نظّف المزرعة من السعف اليابس والمخلفات.",
                           "اليوم 7: دوّن النتائج واسأل المختص عن موعد المتابعة."],
                 "check": "هل خلت النخيل المجاورة من العلامات، وهل المصائد في أماكنها؟"},
                {"focus": "إعادة التقييم", "goal": "تقييم المزرعة كلها وتحديد ما بعد الإصابة.",
                 "tasks": ["اليومان 1 و2: أعد تسجيل كل النخيل القريبة وقارن بقراءات الأسبوع الأول.",
                           "اليوم 3: افحص المصائد وسجّل ما فيها.",
                           "اليوم 4: لا تزرع بديلًا في مكان النخلة قبل مشورة المختص.",
                           "اليوم 5: أرسل ملخصًا بكل ما تم للمختص ليحدد المتابعة.",
                           "اليومان 6 و7: حدّث سجل المزرعة وضع موعدًا لفحص دوري كل أسبوعين."],
                 "check": "هل ثبتت حالة النخيل المجاورة، وهل عندك خطة متابعة مكتوبة مع المختص؟"}],
            "cost": ["الإصابة المتقدمة قد تنتهي بخسارة النخلة، وقد يكلّف تأخير القرار نخيلًا مجاورة أيضًا.",
                     "قد يسقط القلب أو يميل الجذع، وفي ذلك خطر على من حولها، فالتصرف العاجل يحمي الناس كذلك.",
                     "إزالة نخلة واحدة بسرعة أرخص عادةً من خسارة عدة نخلات بعد انتشار السوسة."],
            "watch": ["انتشار الأعراض إلى نخلة ثانية.", "سقوط القلب أو ميلان الجذع: خطر على من حولها.", "تأخر وصول المختص: أعد الاتصال ولا تنتظر."]},
        "en": {
            "headline": {"advanced": "An advanced infestation is possible. It needs urgent action: confirm in the field within days and decide with a specialist between treatment and removal."},
            "meaning": ["The device heard dense, repeated pulses like the activity of many larvae inside the trunk, and the model rated it a possible advanced infestation.",
                        "These are signs, not a diagnosis: the model was trained on simulated recordings and has not been tested on real palms, so urgent field confirmation is needed.",
                        "With an advanced infestation the decision may be between treatment and removal, and the specialist decides after the inspection."],
            "today": ["Call the agricultural extension service or the Ministry of Environment, Water and Agriculture today and ask for an urgent inspection.", "Isolate the palm and do not move any part of it or its offshoots.",
                      "Keep people and children away from the palm if the crown looks weak or the trunk leans.", "Do not treat it yourself with unlicensed products."],
            "weeks": [
                {"focus": "Urgent action", "goal": "Get the specialist's decision fast and isolate the palm so the weevil does not spread.",
                 "tasks": ["Day 1: call the specialist and ask for an urgent inspection, and call again the next day if nobody comes.",
                           "Day 1: isolate the palm, keep visitors and machines away, and do not move fronds, offshoots or debris from it.",
                           "Days 1-2: look over all the nearby palms (holes, brown fluid, sawdust, wilting crown) and record them with the device.",
                           "Day 3: write down the palm's place, photograph it and prepare a spot the specialist and machines can reach.",
                           "Days 4-7: receive the specialist and let them decide: treat or remove, and what each decision needs from you."],
                 "check": "Did the specialist come and decide in writing between treatment and removal, and were all the nearby palms checked?"},
                {"focus": "Treat or remove", "goal": "Carry out the specialist's decision safely and in a way that stops the insect spreading.",
                 "tasks": ["Day 1: fix with the specialist the date and who carries it out, and clear people from around the palm.",
                           "Days 2-4: if it can be saved, follow the treatment plan exactly as the specialist sets it, and use no products of your own.",
                           "Days 2-4: if removal is decided, it is done under the specialist's supervision and the palm is destroyed the way they say so the insect does not spread.",
                           "Day 5: do not leave any trunk, fronds or debris from the palm on the farm or move them elsewhere.",
                           "Days 6-7: write down what was done and when, and keep the specialist's recommendations."],
                 "check": "Was the specialist's decision carried out fully, and was the waste destroyed or treated as asked?"},
                {"focus": "Protect the neighbours", "goal": "Stop the infestation moving to the palms around it.",
                 "tasks": ["Days 1-2: check the neighbouring palms with the device and by eye, recording each palm twice.",
                           "Day 3: do the preventive treatment the specialist recommends for the nearby palms.",
                           "Days 4-5: set up pheromone traps as the specialist advises and note where they are.",
                           "Day 6: clean the farm of dry fronds and debris.",
                           "Day 7: write down the results and ask the specialist when to follow up."],
                 "check": "Are the neighbouring palms free of signs, and are the traps in place?"},
                {"focus": "Review", "goal": "Assess the whole farm and decide what comes after the infestation.",
                 "tasks": ["Days 1-2: record all the nearby palms again and compare with week 1.",
                           "Day 3: check the traps and write down what is in them.",
                           "Day 4: do not replant in the palm's spot before asking the specialist.",
                           "Day 5: send the specialist a summary of everything done so they can set the follow-up.",
                           "Days 6-7: update the farm record and set a regular check every two weeks."],
                 "check": "Is the state of the neighbouring palms stable, and do you have a written follow-up plan with the specialist?"}],
            "cost": ["An advanced infestation may end with losing the palm, and a delayed decision can cost neighbouring palms too.",
                     "A crown may fall or a trunk may lean, which is a danger to people nearby, so acting fast also protects people.",
                     "Removing one palm quickly is usually cheaper than losing several after the weevil spreads."],
            "watch": ["Signs spreading to a second palm.", "A falling crown or a leaning trunk: a danger to people nearby.", "The specialist is late: call again and do not wait."]},
        "ur": {
            "headline": {"advanced": "شدید حملے کا امکان۔ فوری کارروائی چاہیے: چند دنوں میں میدان میں تصدیق کریں اور ماہر کے ساتھ علاج یا ہٹانے کا فیصلہ کریں۔"},
            "meaning": ["ڈیوائس نے تنے کے اندر بہت سی سنڈیوں کی سرگرمی جیسی گھنی اور بار بار آنے والی پلس سنیں، اور ماڈل نے اسے ممکنہ شدید حملہ قرار دیا۔",
                        "یہ علامات ہیں، تشخیص نہیں: ماڈل سمولیشن ریکارڈنگز پر تربیت یافتہ ہے اور حقیقی کھجوروں پر آزمایا نہیں گیا، اس لیے فوری میدانی تصدیق ضروری ہے۔",
                        "شدید حملے میں فیصلہ علاج یا ہٹانے کے درمیان ہو سکتا ہے، اور معائنے کے بعد ماہر طے کرتا ہے۔"],
            "today": ["آج ہی زرعی توسیعی سروس یا وزارتِ ماحولیات، پانی و زراعت کو فون کر کے فوری معائنہ مانگیں۔", "کھجور کو الگ کریں اور اس کا یا اس کے پودوں کا کوئی حصہ منتقل نہ کریں۔",
                      "چوٹی کمزور لگے یا تنا جھکا ہو تو لوگوں اور بچوں کو کھجور سے دور رکھیں۔", "غیر منظور شدہ چیزوں سے خود علاج نہ کریں۔"],
            "weeks": [
                {"focus": "فوری کارروائی", "goal": "ماہر کا فیصلہ جلد حاصل کرنا اور کھجور کو الگ کرنا تاکہ کیڑا نہ پھیلے۔",
                 "tasks": ["دن 1: ماہر کو فون کر کے فوری معائنہ مانگیں، اور کوئی نہ آئے تو اگلے دن دوبارہ فون کریں۔",
                           "دن 1: کھجور کو الگ کریں، زائرین اور مشینوں کو دور رکھیں، اور اس کے پتے، پودے یا کچرا منتقل نہ کریں۔",
                           "دن 1 اور 2: قریب کی تمام کھجوریں دیکھیں (سوراخ، بھورا مائع، برادہ، چوٹی کا مرجھانا) اور ڈیوائس سے ریکارڈ کریں۔",
                           "دن 3: کھجور کی جگہ لکھیں، تصویر لیں اور ایسی جگہ تیار کریں جہاں ماہر اور مشینیں پہنچ سکیں۔",
                           "دن 4 سے 7: ماہر کو آنے دیں اور اسے فیصلہ کرنے دیں: علاج یا ہٹانا، اور ہر فیصلے میں آپ سے کیا چاہیے۔"],
                 "check": "کیا ماہر آیا اور اس نے علاج یا ہٹانے کا فیصلہ تحریری طور پر کیا، اور کیا قریب کی تمام کھجوریں جانچی گئیں؟"},
                {"focus": "علاج یا ہٹانا", "goal": "ماہر کے فیصلے پر محفوظ طریقے سے اور کیڑے کو پھیلنے سے روکتے ہوئے عمل کرنا۔",
                 "tasks": ["دن 1: ماہر کے ساتھ عمل کی تاریخ اور کرنے والا طے کریں، اور کھجور کے گرد سے لوگوں کو ہٹائیں۔",
                           "دن 2 سے 4: بچائی جا سکتی ہو تو علاج کا منصوبہ بالکل ویسے کریں جیسا ماہر طے کرے، اور اپنی طرف سے کچھ استعمال نہ کریں۔",
                           "دن 2 سے 4: ہٹانا طے ہو تو ماہر کی نگرانی میں ہٹائیں اور اس کے بتائے طریقے سے ختم کریں تاکہ کیڑا نہ پھیلے۔",
                           "دن 5: کھجور کا کوئی تنا، پتے یا کچرا باغ میں نہ چھوڑیں اور کہیں اور نہ لے جائیں۔",
                           "دن 6 اور 7: جو کیا اور کب کیا لکھ لیں اور ماہر کی سفارشات سنبھال کر رکھیں۔"],
                 "check": "کیا ماہر کا فیصلہ مکمل طور پر نافذ ہوا، اور کیا کچرا کہنے کے مطابق ختم یا علاج کیا گیا؟"},
                {"focus": "پاس کی کھجوروں کا تحفظ", "goal": "حملے کو آس پاس کی کھجوروں تک جانے سے روکنا۔",
                 "tasks": ["دن 1 اور 2: پاس کی کھجوریں ڈیوائس سے اور نظر سے جانچیں، ہر ایک کو دو بار ریکارڈ کریں۔",
                           "دن 3: قریب کی کھجوروں کے لیے ماہر کا تجویز کردہ احتیاطی علاج کریں۔",
                           "دن 4 اور 5: ماہر کے مشورے سے فیرومون پھندے لگائیں اور ان کی جگہیں لکھیں۔",
                           "دن 6: باغ سے سوکھے پتے اور کچرا صاف کریں۔",
                           "دن 7: نتائج لکھیں اور ماہر سے اگلی جانچ کا وقت پوچھیں۔"],
                 "check": "کیا پاس کی کھجوریں علامات سے خالی ہیں، اور کیا پھندے اپنی جگہ پر ہیں؟"},
                {"focus": "جائزہ", "goal": "پورے باغ کا جائزہ لینا اور حملے کے بعد کا فیصلہ کرنا۔",
                 "tasks": ["دن 1 اور 2: قریب کی تمام کھجوریں دوبارہ ریکارڈ کریں اور پہلے ہفتے سے موازنہ کریں۔",
                           "دن 3: پھندے دیکھیں اور جو ان میں ہو لکھیں۔",
                           "دن 4: ماہر سے پوچھے بغیر کھجور کی جگہ نیا پودا نہ لگائیں۔",
                           "دن 5: جو کچھ ہوا اس کا خلاصہ ماہر کو بھیجیں تاکہ وہ فالو اپ طے کرے۔",
                           "دن 6 اور 7: باغ کا ریکارڈ اپ ڈیٹ کریں اور ہر دو ہفتے بعد باقاعدہ جانچ طے کریں۔"],
                 "check": "کیا پاس کی کھجوروں کی حالت مستحکم ہے، اور کیا ماہر کے ساتھ تحریری فالو اپ منصوبہ آپ کے پاس ہے؟"}],
            "cost": ["شدید حملے کا انجام کھجور کے نقصان پر ہو سکتا ہے، اور فیصلے میں تاخیر پاس کی کھجوروں کو بھی نقصان پہنچا سکتی ہے۔",
                     "چوٹی گر سکتی ہے یا تنا جھک سکتا ہے، جو قریب کے لوگوں کے لیے خطرہ ہے، اس لیے فوری کارروائی لوگوں کو بھی بچاتی ہے۔",
                     "ایک کھجور کو جلد ہٹانا عموماً کیڑا پھیلنے کے بعد کئی کھجوریں کھونے سے سستا ہے۔"],
            "watch": ["علامات دوسری کھجور تک پھیلنا۔", "چوٹی کا گرنا یا تنے کا جھکنا: قریب کے لوگوں کے لیے خطرہ۔", "ماہر دیر کرے تو دوبارہ فون کریں، انتظار نہ کریں۔"]},
    },
}

AI_NAME = {"ar": "حارس", "en": "Haris", "ur": "حارس"}


def _group(stage):
    return "healthy" if stage == "healthy" else ("advanced" if stage == "advanced" else "infested")


def ask_question(language, stage, n, focus):
    """The ready question that opens the assistant for week n (kept under 300 characters: the public assistant cuts there)."""
    t = TITLES[language]
    if language == "ar":
        q = f"نتيجة نخلتي: {t['stage'][stage]}. اشرح لي بالتفصيل الأسبوع {n} ({focus}) من خطة الشهر: ماذا أفعل يومًا بيوم وما الذي أنتبه له؟"
    elif language == "ur":
        q = f"میری کھجور کا نتیجہ: {t['stage'][stage]}۔ مجھے مہینے کے منصوبے کے ہفتہ {n} ({focus}) کی تفصیل بتائیں: دن بہ دن کیا کروں اور کس بات کا خیال رکھوں؟"
    else:
        q = f"My palm's result: {t['stage'][stage]}. Explain in detail week {n} ({focus}) of the month plan: what do I do day by day and what do I watch for?"
    return q[:290]


def ask_question_all(language, stage):
    t = TITLES[language]
    if language == "ar":
        q = f"نتيجة نخلتي: {t['stage'][stage]}. لخّص لي الخطوات الأهم في خطة هذا الشهر، وما الذي أطلبه من المختص؟"
    elif language == "ur":
        q = f"میری کھجور کا نتیجہ: {t['stage'][stage]}۔ اس مہینے کے منصوبے کے اہم ترین قدم بتائیں، اور ماہر سے مجھے کیا مانگنا چاہیے؟"
    else:
        q = f"My palm's result: {t['stage'][stage]}. Sum up the most important steps of this month's plan, and what should I ask the specialist for?"
    return q[:290]


def _cards(language, stage, meaning, today, weeks, cost, watch):
    """weeks: list of dicts {focus, goal, tasks, check}"""
    t = TITLES[language]
    cards = [{"id": "meaning", "title": t["meaning"], "items": meaning}, {"id": "today", "title": t["today"], "items": today}]
    for i, w in enumerate(weeks, 1):
        card = {"id": f"w{i}", "group": "plan", "title": t[f"w{i}"], "focus": w["focus"], "items": w["tasks"], "ask": ask_question(language, stage, i, w["focus"])}
        if w.get("goal"):
            card["goal"] = w["goal"]
        if w.get("check"):
            card["check"] = w["check"]
        cards.append(card)
    cards.append({"id": "cost", "title": t["cost"], "items": cost})
    cards.append({"id": "watch", "title": t["watch"], "items": watch})
    return cards


def _labels(language):
    t = TITLES[language]
    return {"check": t["check"], "goal": t["goal"], "ask_btn": t["ask_btn"], "ask_all": t["ask_all"]}


def template_report(stage, language):
    language = language if language in TITLES else "en"
    stage = stage if stage in STAGES else "healthy"
    c = _T[_group(stage)][language]
    return {"source": "template", "ai_name": AI_NAME[language], "headline": c["headline"].get(stage) or next(iter(c["headline"].values())),
            "plan_title": TITLES[language]["plan"], "labels": _labels(language), "ask_all": ask_question_all(language, stage),
            "cards": _cards(language, stage, c["meaning"], c["today"], c["weeks"], c["cost"], c["watch"]), "note": TITLES[language]["note"]}


# ---------------------------------------------------------------- the AI version
_LANG_NAME = {"ar": "Arabic", "en": "English", "ur": "Urdu"}


def build_prompt(stage, risk, evidence, language):
    lang = _LANG_NAME.get(language, "English")
    system = (
        "You are Haris, the assistant of NakhlaGuard, a prototype that listens to a date palm's trunk to look for red palm weevil larvae. "
        f"Write a complete, practical treatment report for a farmer, in {lang}. Answer with ONE JSON object and nothing else, with exactly these keys: "
        '"headline" (one sentence, max 160 characters), '
        '"meaning" (2 to 3 sentences: what this result means in plain words, and that it must be confirmed in the field), '
        '"today" (3 to 4 action sentences for today), '
        '"weeks" (exactly 4 objects, week 1 to week 4, each {"focus": 2 to 4 words, "goal": one sentence, '
        '"tasks": 4 to 6 concrete actions in time order, each starting with the day, for example "Day 1: ..." or "Days 3-4: ...", '
        '"check": one sentence saying what the farmer checks at the end of that week}), '
        '"cost" (2 to 3 sentences on why acting early matters, in words only, with no invented numbers or percentages), '
        '"call_expert" (3 to 4 short warning signs that mean call a specialist now). '
        "Every sentence is at most 200 characters. Plan one month in a sensible order for the stage (confirm and contain, treat, follow up, review; "
        "for a healthy result: baseline, farm hygiene, traps and advice, re-check). Be specific about what to look at, where, how often and what to write down, "
        "so the farmer knows exactly what to do each day. "
        "General guidance only: never name pesticide brands or products, never give doses or quantities of chemicals, never promise a cure. Always leave the confirmation and the treatment "
        "decision to a specialist (the agricultural extension service or the Ministry of Environment, Water and Agriculture). "
        "The result came from a model trained on simulated recordings, so say the farmer must confirm in the field. "
        "Say 'palm', never 'tree'. No markdown, no emojis."
    )
    ev = evidence or {}
    user = (f"Result: stage = {stage}"
            + (f", risk = {round(float(risk) * 100)}%" if risk is not None else "")
            + f". Short pulses heard: {ev.get('pulses', 'unknown')} in {ev.get('duration_s', 'unknown')} s. "
              "Write the report.")
    return system, user


def _clean_list(v, lo, hi, maxlen=230):
    if not isinstance(v, list):
        return None
    out = []
    for x in v:
        if isinstance(x, str) and x.strip():
            out.append(re.sub(r"\s+", " ", x.strip())[:maxlen])
    return out[:hi] if len(out) >= lo else None


def _clean_str(v, maxlen=240):
    return re.sub(r"\s+", " ", v.strip())[:maxlen] if isinstance(v, str) and v.strip() else None


def parse_ai_json(text):
    """Returns {"headline", "meaning" (or None), "today", "weeks": [{focus, goal, tasks, check}], "cost" (or None), "watch"} or None if the answer is not usable.
    The minimum a usable answer needs: headline, today, call_expert, and four weeks with a focus and at least 2 tasks. "meaning" and "cost" are optional
    (the ready-made ones are used if the AI leaves them out)."""
    if not isinstance(text, str):
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    headline = _clean_str(d.get("headline"), 200)
    today = _clean_list(d.get("today"), 2, 5)
    watch = _clean_list(d.get("call_expert"), 2, 5)
    weeks_raw = d.get("weeks")
    if not (headline and today and watch and isinstance(weeks_raw, list) and len(weeks_raw) >= 4):
        return None
    weeks = []
    for w in weeks_raw[:4]:
        if not isinstance(w, dict) or not isinstance(w.get("focus"), str):
            return None
        tasks = _clean_list(w.get("tasks"), 2, 7)
        if not tasks:
            return None
        weeks.append({"focus": _clean_str(w["focus"], 60), "goal": _clean_str(w.get("goal")), "tasks": tasks, "check": _clean_str(w.get("check"))})
    return {"headline": headline, "meaning": _clean_list(d.get("meaning"), 1, 4), "today": today, "weeks": weeks,
            "cost": _clean_list(d.get("cost"), 1, 4), "watch": watch}


async def make_report(stage, risk, evidence, language, ask_ai):
    """ask_ai(system, user) -> text (async). It may raise or return None; then the ready-made report is used.
    Returns the dict the page renders; "source" says "ai" or "template"."""
    language = language if language in TITLES else "en"
    stage = stage if stage in STAGES else "healthy"
    base = template_report(stage, language)
    if ask_ai is None:
        return base
    system, user = build_prompt(stage, risk, evidence, language)
    try:
        # a judge must not stare at a spinner: after this many seconds the ready-made (labelled) report is shown instead
        p = parse_ai_json(await asyncio.wait_for(ask_ai(system, user), timeout=AI_REPORT_TIMEOUT_S))
    except Exception as e:   # includes asyncio.TimeoutError
        print(f"[NakhlaGuard demo] AI report failed: {type(e).__name__}: {e}")
        p = None
    if not p:
        return base
    c = _T[_group(stage)][language]
    return {"source": "ai", "ai_name": AI_NAME[language], "headline": p["headline"], "plan_title": TITLES[language]["plan"], "labels": _labels(language),
            "ask_all": ask_question_all(language, stage),
            "cards": _cards(language, stage, p["meaning"] or c["meaning"], p["today"], p["weeks"], p["cost"] or c["cost"], p["watch"]),
            "note": TITLES[language]["note"]}
