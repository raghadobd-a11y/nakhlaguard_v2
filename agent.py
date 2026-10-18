"""
وكيل نخلة الذكي (Agentic AI) - مشروع NakhlaGuard
=================================================
الفرق بين هذا وبين استدعاء Gemini العادي بـ fusion.py:
fusion.py يرسل سؤال واحد ويستقبل جواب واحد (نمط "استدعاء" بسيط).

هذا الملف وكيل حقيقي (Agent): يعطى الوصول لأدوات حقيقية تقرأ من
قاعدة بيانات المزرعة، ويقرر بنفسه (بدون مسار مبرمج مسبقًا):
- هل يحتاج بيانات فعلية للإجابة، ولو يحتاج، أي أداة يستخدم؟
- هل جواب أداة وحدة كافي، ولا يحتاج يستدعي أداة ثانية؟
- متى يتوقف ويعطي الجواب النهائي للمزارع؟

هذا نمط "Tool Use / Function Calling Loop" المعياري لبناء وكلاء
ذكاء اصطناعي، مبني على generateContent API نفسه عبر عميل مشترك
(gemini_client.py) يعيد المحاولة تلقائيًا عند ازدحام خادم Gemini
المؤقت (503/429) - بدون الحاجة لأي مكتبة خارجية إضافية.
"""
import os
from gemini_client import call_gemini
from severity_labels import anglicize

AGENT_SYSTEM_PROMPT_EN = """You are "Farm Assistant" - a highly knowledgeable AI \
agent inside the NakhlaGuard system, specialized in Red Palm Weevil (RPW) \
management for date palm farmers. You draw on deep domain knowledge of RPW \
biology, risk factors, and integrated pest management, and you answer with the \
confidence and precision of a seasoned specialist - while always grounding \
farm-specific claims in real data (never guessing numbers or locations).

Key domain knowledge you should apply naturally when relevant:
- Age is a major risk factor. Palms aged 3-15 years (peak risk 5-10 years) are \
most vulnerable: their trunk tissue is soft, moist, and sugar-rich (ideal larval \
food), and ground-level offshoots/frond bases offer easy egg-laying spots. \
Newly cut offshoots (under 3 years) are also high-risk because the fresh cut \
wound attracts egg-laying females. Palms over 20 years are more resistant since \
the wood hardens and lignifies - if infested, it is usually at the crown \
(jummar) rather than the trunk base.
- A farm with many young palms (under 15 years) needs more frequent inspection \
and preventive measures (base treatment, pheromone traps), avoiding trunk \
wounds except in cold weather.

You have real tools that connect you to the actual farm data:
- list_palms: every registered palm, its location, age, and its latest check status
- get_palm_details: full details for one specific palm (location, notes, age, and \
its complete check history over time)
- palms_needing_attention: only the palms whose current status is moderate or \
advanced (need the farmer's attention now)
- palms_due_for_recheck: palms not re-checked in 30+ days - a general preventive \
reminder, not tied to any specific disease season
- get_farm_contact: the farm's backup contact (name, phone) to reach when the \
owner is unavailable

Core rule: any question about real farm data (how many palms, where is palm X, \
what's the status of X, which palms need attention, how old is palm X) - you \
must call the right tool before answering. Never guess or invent numbers or \
locations. General RPW knowledge questions (like the age-risk facts above) can \
be answered directly without a tool.

Respond in English only, regardless of the language the farmer wrote in - be \
concise, clear, and practically useful."""

AGENT_SYSTEM_PROMPT_AR = """أنت "مساعد نخلة" - وكيل ذكاء اصطناعي واسع المعرفة
داخل نظام NakhlaGuard، متخصص بإدارة آفة سوسة النخيل الحمراء لمزارعي النخيل.
تعتمد على معرفة عميقة بأحياء الآفة، عوامل الخطورة، والمكافحة المتكاملة،
وتجيب بثقة ودقة أخصائي متمرّس - مع تأكيد أي معلومة خاصة بمزرعة معينة عبر
بيانات حقيقية دائمًا (بدون تخمين أرقام أو مواقع أبدًا).

معرفة أساسية تطبّقها بشكل طبيعي متى ما كانت مناسبة:
- العمر عامل خطورة رئيسي. النخيل بعمر 3-15 سنة (وذروة الخطورة بين 5-10)
  الأكثر عرضة: أنسجة جذعها طرية ورطبة وغنية بالسكريات (غذاء مثالي
  لليرقات)، والفسائل وقواعد السعف القريبة من الأرض توفر مخابئ سهلة لوضع
  البيض. الفسائل المفصولة حديثًا (أقل من 3 سنوات) عالية الخطورة أيضًا
  لأن جرح الفصل الطازج يجذب إناث وضع البيض. النخيل أكبر من 20 سنة أكثر
  مقاومة لأن الخشب يتصلّب ويتلجّن - ولو أُصيب، غالبًا يكون بالجُمّارة
  (القمة النامية) لا بقاعدة الجذع.
- المزرعة اللي فيها نخيل شاب كثير (أقل من 15 سنة) تحتاج فحصًا دوريًا
  أكثر تكرارًا وإجراءات وقائية (رش القواعد، مصائد الفيرومون)، وتجنّب
  جروح الجذع إلا بالطقس البارد.

لديك أدوات حقيقية تصلك ببيانات المزرعة الفعلية:
- list_palms: كل النخيل المسجَّلة، مواقعها، أعمارها، وحالة آخر فحص لكل واحدة
- get_palm_details: التفاصيل الكاملة لنخلة معينة (الموقع، الملاحظات، العمر،
  وتاريخ جميع الفحوصات التي أُجريت لها)
- palms_needing_attention: فقط النخيل التي حالتها الحالية متوسطة أو
  متقدمة (تحتاج إلى تدخل المزارع الآن)
- palms_due_for_recheck: النخيل التي لم تُفحص من 30 يومًا فأكثر - تذكير
  وقائي عام، غير مرتبط بموسم مرض معيّن
- get_farm_contact: جهة الاتصال الاحتياطية للمزرعة (الاسم والجوال) اللي
  يُتواصل معها إذا كان صاحب المزرعة غير متوفر

قاعدة أساسية: أي سؤال عن بيانات فعلية بالمزرعة (عدد النخيل، موقع نخلة
معينة، حالة نخلة معينة، عمر نخلة معينة، أي نخلة تحتاج إلى عناية) - يجب
استخدام الأداة المناسبة قبل الإجابة. لا تخمّن أو تختلق أرقامًا أو مواقع
أبدًا. أسئلة المعرفة العامة عن السوسة (زي معلومات خطورة العمر أعلاه)
تقدر تجاوب عليها مباشرة بدون أداة.

أجب بالعربية الفصحى المبسطة فقط، بغض النظر عن لغة سؤال المزارع - بإيجاز
ووضوح وأسلوب عملي يفيده فعليًا."""

TOOLS = [{
    "function_declarations": [
        {
            "name": "list_palms",
            "description": (
                "Get every registered palm tree with its location and its "
                "most recent health status (healthy / early / moderate / advanced)."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "get_palm_details",
            "description": (
                "Get full details for one specific palm tree: its location, "
                "notes, and its complete check history over time (every "
                "past reading with date and severity)."
            ),
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "palm_id": {
                        "type": "STRING",
                        "description": "The palm's ID exactly as registered, e.g. Palm-A12",
                    },
                },
                "required": ["palm_id"],
            },
        },
        {
            "name": "palms_needing_attention",
            "description": (
                "Get only the palms whose latest check came back moderate "
                "or advanced infestation - i.e. palms that currently need "
                "the farmer's attention right now."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "palms_due_for_recheck",
            "description": (
                "Get the palms that have not been re-checked in 30 days or "
                "more (or never checked since registration) - a general "
                "preventive reminder list, independent of current severity "
                "and not tied to any specific disease season."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
        {
            "name": "get_farm_contact",
            "description": (
                "Get the farm's registered backup contact (name and phone) - "
                "the person to reach when the farm owner is unavailable "
                "(traveling, out of the country) so urgent alerts still "
                "reach someone who can act on them."
            ),
            "parameters": {"type": "OBJECT", "properties": {}},
        },
    ]
}]


async def run_agent(user_message, tool_executor, history=None, max_steps=5, language="en"):
    """
    حلقة الوكيل (Agent Loop):
    1. يرسل رسالة المزارع لـ Gemini مع تعريف الأدوات المتاحة، بنظام
       التعليمات المناسب للغة المطلوبة (language: "en" أو "ar")
    2. لو النموذج قرر يستدعي أداة (functionCall) -> ننفذها فعليًا
       بقاعدة البيانات الحقيقية، ونرجّع نتيجتها له (functionResponse)
    3. يكرر هذا لين النموذج يوصل لجواب نهائي (نص) أو نوصل للحد الأقصى
       من الخطوات (max_steps) كحماية من التكرار اللانهائي

    tool_executor: dict {اسم_الأداة: دالة بايثون فعلية تنفذها} - لازم
    نفس الأسماء المعرّفة بـ TOOLS بالضبط.
    history: قائمة رسائل سابقة بصيغة contents (لمحادثة متعددة الأدوار).

    يرجّع: {
        "answer": الجواب النهائي كنص,
        "trace": [{"tool": اسم_الأداة, "args": المدخلات}, ...] بترتيب
                 الاستدعاء - مفيد لعرض "كيف فكّر الوكيل" بالعرض التقديمي
    }
    """
    system_prompt = AGENT_SYSTEM_PROMPT_AR if language == "ar" else AGENT_SYSTEM_PROMPT_EN

    if language == "ar":
        no_key_msg = "⚠️ لم يتم ضبط GEMINI_API_KEY - المساعد الذكي يحتاج المفتاح ليعمل."
        fail_prefix = "⚠️ تعذّر الاتصال بالمساعد الذكي حاليًا"
        stuck_msg = "⚠️ الوكيل استخدم عدة أدوات بدون التوصل لجواب نهائي - جربي إعادة صياغة السؤال."
    else:
        no_key_msg = "⚠️ GEMINI_API_KEY is not set - the AI assistant needs it to work."
        fail_prefix = "⚠️ Could not reach the AI assistant right now"
        stuck_msg = "⚠️ The agent used several tools without reaching a final answer - try rephrasing your question."

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {"answer": no_key_msg, "trace": []}

    contents = list(history) if history else []
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    trace = []

    for _ in range(max_steps):
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "tools": TOOLS,
        }
        try:
            data = await call_gemini(payload, api_key)
        except Exception as e:
            print(f"[NakhlaGuard Agent] call failed: {type(e).__name__}: {e}")
            return {"answer": f"{fail_prefix} ({type(e).__name__}).", "trace": trace}

        candidate = data["candidates"][0]["content"]
        parts = candidate.get("parts", [])
        function_call_part = next((p for p in parts if "functionCall" in p), None)

        if function_call_part is None:
            # النموذج ما طلب أي أداة - هذا جوابه النهائي
            final_text = "".join(p.get("text", "") for p in parts)
            return {"answer": final_text, "trace": trace}

        # النموذج قرر بنفسه يستدعي أداة - ننفذها فعليًا على بياناتنا
        fn_name = function_call_part["functionCall"]["name"]
        fn_args = function_call_part["functionCall"].get("args", {}) or {}
        trace.append({"tool": fn_name, "args": fn_args})

        contents.append({"role": "model", "parts": [function_call_part]})

        executor = tool_executor.get(fn_name)
        if executor is None:
            tool_result = {"error": f"Unknown tool: {fn_name}"}
        else:
            try:
                tool_result = executor(**fn_args)
            except Exception as e:
                tool_result = {"error": str(e)}

        # بالإنجليزي: نترجم أي تصنيف شدة عربي (سليم/مبكرة...) للإنجليزي
        # قبل ما يوصل النموذج. بالعربي: نبقيها كما هي (أطبع لجواب عربي).
        if language == "en":
            tool_result = anglicize(tool_result)

        contents.append({
            "role": "user",
            "parts": [{
                "functionResponse": {
                    "name": fn_name,
                    "response": {"result": tool_result},
                }
            }],
        })

    return {"answer": stuck_msg, "trace": trace}
