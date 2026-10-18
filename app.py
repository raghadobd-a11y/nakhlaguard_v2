"""
تطبيق NakhlaGuard - كشف مبكر لإصابة النخيل بسوسة النخيل الحمراء
================================================================
نفس نمط مشروعك بصيرة: FastAPI + Gemini + واجهة ويب بسيطة.

ملاحظة هيكلية: كل ملفات المشروع (بايثون وHTML) بمستوى واحد بدون
مجلدات فرعية (عشان تتوافق مع الرفع المباشر على GitHub عبر السحب
والإفلات بدون مشاكل مجلدات). كل صفحة HTML تُخدَّم مباشرة من الجذر
(/index.html بدل /static/index.html).

المسارات:
- GET  /                  : الصفحة الرئيسية (index.html)
- GET  /<اسم>.html        : أي صفحة ويب بالمشروع (test, dashboard, palms, agent)
- POST /predict-audio     : يرفع تسجيل صوتي -> شدة الإصابة + الثقة
- POST /predict-full      : صوت (+ صورة اختياري + موقع) -> دمج + توصية Gemini
                            (تقرير كامل تلقائيًا لو الحالة تستدعي عناية)
                            + إشعار فوري تلقائي لو الحالة متوسطة/متقدمة
- GET  /history/{palm_id} : سجل فحوصات نخلة معينة (لرسم منحنى التطور)
- GET  /palms             : سجل كل النخيل المسجّلة + مواقعها + آخر حالة لكل وحدة
- POST /palms             : تسجيل/تحديث نخلة (رقمها + موقعها + ملاحظات)
- WS   /ws/alerts         : اتصال مباشر (WebSocket) - لوحة المزارع تستقبل
                            عليه إشعارات لحظية بمجرد اكتشاف إصابة، بدون
                            حاجة لتحديث الصفحة يدويًا
- POST /agent/chat        : وكيل ذكاء اصطناعي (Agentic AI) - يجاوب أسئلة
                            المزارع عن نخيله عبر استدعاء أدوات حقيقية على
                            قاعدة البيانات (مو نص مبرمج مسبقًا) - شوفي agent.py
"""
import os
import json
import tempfile
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# كل الملفات بنفس مجلد app.py، فـ Python يضيف هذا المجلد لمسار
# البحث تلقائيًا - ما نحتاج أي sys.path.append إضافي
from predict import predict_severity_from_file  # noqa: E402
from fusion import fuse_audio_image, get_gemini_recommendation, estimate_age_and_health_from_image  # noqa: E402
from agent import run_agent  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nakhla.db")

app = FastAPI(title="NakhlaGuard API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


def _html(filename):
    """يرجّع صفحة HTML من نفس مجلد المشروع (بدون الحاجة لمجلد static/ منفصل)."""
    return FileResponse(os.path.join(BASE_DIR, filename))


@app.get("/")
def serve_home():
    return _html("index.html")


@app.get("/index.html")
def serve_index():
    return _html("index.html")


@app.get("/test.html")
def serve_test():
    return _html("test.html")


@app.get("/dashboard.html")
def serve_dashboard():
    return _html("dashboard.html")


@app.get("/palms.html")
def serve_palms_page():
    return _html("palms.html")


@app.get("/agent.html")
def serve_agent_page():
    return _html("agent.html")


@app.get("/instructions.html")
def serve_instructions_page():
    return _html("instructions.html")


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            palm_id TEXT NOT NULL,
            predicted_class TEXT,
            final_score REAL,
            created_at TEXT
        )
    """)
    # سجل النخيل: كل نخلة مسجّلة بموقعها (نص حر - رقم صف، إحداثيات،
    # أي وصف يفهمه المزارع)، عشان لو عنده نخل كثير يلقى كل وحدة بسهولة
    conn.execute("""
        CREATE TABLE IF NOT EXISTS palms (
            palm_id TEXT PRIMARY KEY,
            location TEXT,
            notes TEXT,
            age_years INTEGER,
            created_at TEXT
        )
    """)
    # ترقية آمنة لقاعدة بيانات قديمة ما فيها عمود العمر بعد (لو الجدول
    # كان موجود قبل ما نضيف هالميزة) - نتجاهل الخطأ لو العمود موجود أصلاً
    try:
        conn.execute("ALTER TABLE palms ADD COLUMN age_years INTEGER")
    except sqlite3.OperationalError:
        pass

    # جهة اتصال احتياطية واحدة للمزرعة كاملة (مو لكل نخلة لحالها) - تبان
    # بكل تنبيه خطر، عشان لو صاحب المزرعة مو متوفر (مسافر مثلاً)، فيه
    # شخص ثاني (عامل، قريب) يقدر يتواصلون معه يروح يعالج النخلة
    conn.execute("""
        CREATE TABLE IF NOT EXISTS farm_contact (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            contact_name TEXT,
            contact_phone TEXT
        )
    """)
    conn.commit()
    conn.close()


init_db()


def save_reading(palm_id, predicted_class, final_score):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO readings (palm_id, predicted_class, final_score, created_at) VALUES (?,?,?,?)",
        (palm_id, predicted_class, final_score, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_history(palm_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT predicted_class, final_score, created_at FROM readings "
        "WHERE palm_id=? ORDER BY created_at",
        (palm_id,),
    ).fetchall()
    conn.close()
    return [{"class": r[0], "score": r[1], "date": r[2]} for r in rows]


def upsert_palm(palm_id, location=None, notes=None, age_years=None):
    """يسجّل نخلة جديدة أو يحدّث موقعها/ملاحظاتها/عمرها لو كانت مسجّلة مسبقًا."""
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT palm_id FROM palms WHERE palm_id=?", (palm_id,)).fetchone()
    if existing:
        if location is not None:
            conn.execute("UPDATE palms SET location=? WHERE palm_id=?", (location, palm_id))
        if notes is not None:
            conn.execute("UPDATE palms SET notes=? WHERE palm_id=?", (notes, palm_id))
        if age_years is not None:
            conn.execute("UPDATE palms SET age_years=? WHERE palm_id=?", (age_years, palm_id))
    else:
        conn.execute(
            "INSERT INTO palms (palm_id, location, notes, age_years, created_at) VALUES (?,?,?,?,?)",
            (palm_id, location, notes, age_years, datetime.utcnow().isoformat()),
        )
    conn.commit()
    conn.close()


def get_palm_location(palm_id):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT location FROM palms WHERE palm_id=?", (palm_id,)).fetchone()
    conn.close()
    return row[0] if row else None


def list_palms():
    """يرجّع كل النخيل المسجّلة مع آخر حالة فحص لكل وحدة (لعرضها كقائمة/خريطة)."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT p.palm_id, p.location, p.notes, p.age_years, p.created_at,
               r.predicted_class, r.final_score, r.created_at
        FROM palms p
        LEFT JOIN readings r ON r.id = (
            SELECT id FROM readings WHERE palm_id = p.palm_id
            ORDER BY created_at DESC LIMIT 1
        )
        ORDER BY p.created_at DESC
    """).fetchall()
    conn.close()
    return [
        {
            "palm_id": r[0], "location": r[1], "notes": r[2], "age_years": r[3], "registered_at": r[4],
            "latest_status": r[5], "latest_score": r[6], "latest_check": r[7],
        }
        for r in rows
    ]


def get_palm_details(palm_id):
    """أداة الوكيل: تفاصيل نخلة معينة كاملة (موقع، ملاحظات، عمر، تاريخ فحوصات)."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT location, notes, age_years, created_at FROM palms WHERE palm_id=?", (palm_id,)
    ).fetchone()
    conn.close()
    check_history = get_history(palm_id)
    if not row and not check_history:
        return {"error": f"No palm registered with ID '{palm_id}'."}
    return {
        "palm_id": palm_id,
        "location": row[0] if row else None,
        "notes": row[1] if row else None,
        "age_years": row[2] if row else None,
        "registered_at": row[3] if row else None,
        "check_history": check_history,
    }


def palms_needing_attention():
    """أداة الوكيل: النخيل اللي حالتها الحالية متوسطة أو متقدمة بس."""
    return [p for p in list_palms() if p["latest_status"] in ALERT_LEVELS]


# عدد الأيام الافتراضي اللي لو عدّى بدون فحص جديد، النخلة تعتبر "محتاجة
# فحص دوري" - ممارسة وقائية عامة، مو مبنية على موسم مرض معيّن
RECHECK_DUE_DAYS = 30


def palms_due_for_recheck(days=RECHECK_DUE_DAYS):
    """
    أداة الوكيل + لوحة المزارع: النخيل اللي ما انفحصت من (days) يوم أو أكثر
    (أو ما انفحصت أبدًا من وقت تسجيلها). هذا تذكير فحص دوري وقائي عام، مو
    تنبؤًا بموسم إصابة لمرض معيّن.
    """
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    due = []
    for p in list_palms():
        last = p["latest_check"] or p["registered_at"]
        if last is None or last < cutoff:
            due.append({**p, "days_since_last_check": (
                (datetime.utcnow() - datetime.fromisoformat(last)).days if last else None
            )})
    return due


def get_farm_contact():
    """جهة الاتصال الاحتياطية للمزرعة (صف واحد بس) - أو None لو ما سُجّلت."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT contact_name, contact_phone FROM farm_contact WHERE id=1").fetchone()
    conn.close()
    if not row or (not row[0] and not row[1]):
        return None
    return {"contact_name": row[0], "contact_phone": row[1]}


def set_farm_contact(contact_name, contact_phone):
    """يسجّل أو يحدّث جهة الاتصال الاحتياطية (صف واحد ثابت id=1)."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO farm_contact (id, contact_name, contact_phone) VALUES (1, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET contact_name=excluded.contact_name, contact_phone=excluded.contact_phone",
        (contact_name, contact_phone),
    )
    conn.commit()
    conn.close()


# ---------- نظام الإشعارات الفورية (WebSocket) ----------
# الحد الأدنى لشدة الإصابة اللي يستدعي إرسال إشعار للمزارع
ALERT_LEVELS = {"متوسطة", "متقدمة"}

connected_clients: list[WebSocket] = []


@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    لوحة المزارع (dashboard.html) تفتح اتصال هنا وتبقى منتظرة.
    بمجرد ما فحص جديد يطلع بنتيجة خطرة، يوصلها إشعار فوري بدون
    ما تحتاج تحدّث الصفحة أو ترسل طلب بنفسها (Push بدل Polling).
    """
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # نخليه مفتوح، ما نحتاج نستقبل شي فعليًا
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


async def broadcast_alert(payload: dict):
    """يرسل الإشعار لكل الشاشات المفتوحة على لوحة المزارع حاليًا."""
    dead = []
    for client in connected_clients:
        try:
            await client.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            dead.append(client)
    for d in dead:
        connected_clients.remove(d)


@app.post("/predict-audio")
async def predict_audio(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        result = predict_severity_from_file(tmp_path)
    finally:
        os.remove(tmp_path)
    return JSONResponse(result)


@app.post("/predict-full")
async def predict_full(
    audio: UploadFile = File(...),
    palm_id: str = Form(...),
    location: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    image_infested_probability: Optional[float] = Form(None),
    language: str = Form("en"),
    age_years: Optional[int] = Form(None),
    visual_notes: Optional[str] = Form(None),
):
    """
    التحليل الكامل: صوت (إلزامي) + موقع النخلة (اختياري، يُسجَّل تلقائيًا)
    + وصف إضافي (اختياري) + احتمال إصابة من الصورة (اختياري) + عمر النخلة
    (اختياري، يدوي أو مُقدَّر من الصورة) + ملاحظات بصرية صحية عامة (اختياري)
    + لغة الجواب (en/ar) -> دمج + توصية/تقرير كامل + حفظ بالسجل.
    """
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        audio_result = predict_severity_from_file(tmp_path)
    finally:
        os.remove(tmp_path)

    # نسجّل/نحدّث موقع النخلة ووصفها وعمرها تلقائيًا لو انبعثوا معنا - يبني
    # سجل النخيل تدريجيًا بدون ما يحتاج المزارع خطوة تسجيل منفصلة إجبارية
    upsert_palm(palm_id, location=location, notes=notes, age_years=age_years)
    saved_location = location or get_palm_location(palm_id)

    fusion_result = fuse_audio_image(audio_result, image_infested_probability)
    history = get_history(palm_id)
    recommendation = await get_gemini_recommendation(
        fusion_result, palm_id, history, saved_location, language, age_years, visual_notes
    )
    save_reading(palm_id, fusion_result["final_class"], fusion_result["final_score"])

    # إرسال إشعار فوري للمزارع لو الحالة وصلت لمستوى خطورة يستدعي تنبيه
    # - نرفق جهة الاتصال الاحتياطية (لو مسجّلة) عشان لو صاحب المزرعة مو
    #   متوفر، يبان بنفس التنبيه مين يتواصلون معه بدلًا عنه
    if fusion_result["final_class"] in ALERT_LEVELS:
        await broadcast_alert({
            "type": "alert",
            "palm_id": palm_id,
            "location": saved_location,
            "severity": fusion_result["final_class"],
            "score": fusion_result["final_score"],
            "recommendation": recommendation,
            "farm_contact": get_farm_contact(),
            "timestamp": datetime.utcnow().isoformat(),
        })

    return JSONResponse({
        "audio_result": audio_result,
        "fusion_result": fusion_result,
        "recommendation": recommendation,
        "history": get_history(palm_id),
        "location": saved_location,
        "needs_attention": fusion_result["final_class"] in ALERT_LEVELS,
    })


@app.post("/estimate-age-health")
async def estimate_age_health(photo: UploadFile = File(...)):
    """
    ترفع صورة عادية للنخلة (مو تسجيل صوتي) -> Gemini Vision يقدّر عمرها
    التقريبي (من طول/سمك الجذع وكثافة السعف) ويفحص علامات إجهاد/جفاف
    بصرية بسيطة - كل هذا بدون أي حساس إضافي، بس من نفس الكاميرا.
    """
    photo_bytes = await photo.read()
    mime_type = photo.content_type or "image/jpeg"
    result = await estimate_age_and_health_from_image(photo_bytes, mime_type)
    if result is None:
        return JSONResponse(
            {"error": "Could not estimate from this photo right now. You can enter the age manually instead."},
            status_code=200,
        )
    return JSONResponse(result)


@app.get("/history/{palm_id}")
def history(palm_id: str):
    return {"palm_id": palm_id, "readings": get_history(palm_id)}


@app.get("/palms")
def palms():
    """سجل كل النخيل المسجّلة مع موقعها وعمرها وآخر حالة فحص - قائمة سهلة
    يستخدمها المزارع لتتبع نخيله كلها بنظرة وحدة."""
    return {"palms": list_palms()}


@app.get("/palms/due-for-recheck")
def palms_due_for_recheck_route():
    """النخيل اللي تحتاج فحصًا دوريًا (ما انفحصت من 30 يوم أو أكثر) -
    تذكير وقائي عام، مو تنبؤًا بموسم مرض معيّن."""
    return {"palms": palms_due_for_recheck(), "threshold_days": RECHECK_DUE_DAYS}


@app.get("/farm-contact")
def farm_contact_route():
    """جهة الاتصال الاحتياطية للمزرعة (null لو ما سُجّلت بعد)."""
    return get_farm_contact() or {"contact_name": None, "contact_phone": None}


@app.post("/farm-contact")
async def set_farm_contact_route(
    contact_name: str = Form(...),
    contact_phone: str = Form(...),
):
    """يسجّل/يحدّث جهة الاتصال الاحتياطية - تبان بكل تنبيه خطر مستقبلي،
    عشان لو صاحب المزرعة مو متوفر (مسافر مثلًا) فيه شخص ثاني يتواصلون
    معه يروح يعالج النخلة."""
    set_farm_contact(contact_name, contact_phone)
    return {"status": "ok", "contact_name": contact_name, "contact_phone": contact_phone}


@app.post("/palms")
async def register_palm(
    palm_id: str = Form(...),
    location: str = Form(...),
    notes: Optional[str] = Form(None),
    age_years: Optional[int] = Form(None),
):
    """تسجيل نخلة جديدة بموقعها مسبقًا (قبل أول فحص لها)، أو تحديث موقع نخلة موجودة."""
    upsert_palm(palm_id, location=location, notes=notes, age_years=age_years)
    return {"status": "ok", "palm_id": palm_id, "location": location}


# الأدوات الحقيقية اللي الوكيل الذكي يقدر يستدعيها - نفس أسماء
# TOOLS المعرّفة بـ agent.py بالضبط
AGENT_TOOL_EXECUTOR = {
    "list_palms": lambda: list_palms(),
    "get_palm_details": lambda palm_id: get_palm_details(palm_id),
    "palms_needing_attention": lambda: palms_needing_attention(),
    "palms_due_for_recheck": lambda: palms_due_for_recheck(),
    "get_farm_contact": lambda: (get_farm_contact() or {"status": "no backup contact registered yet"}),
}


@app.post("/agent/chat")
async def agent_chat(message: str = Form(...), history: Optional[str] = Form(None), language: str = Form("en")):
    """
    وكيل ذكاء اصطناعي (Agentic AI): يجاوب أسئلة المزارع عن نخيله باستدعاء
    أدوات حقيقية على قاعدة البيانات بنفسه - انظري agent.py للحلقة الكاملة.
    """
    parsed_history = json.loads(history) if history else []
    result = await run_agent(message, AGENT_TOOL_EXECUTOR, parsed_history, language=language)
    return JSONResponse(result)


if __name__ == "__main__":
    import uvicorn
    # Render (وخدمات الاستضافة المشابهة) تحدد رقم المنفذ تلقائيًا عبر
    # متغير بيئة PORT - نستخدمه لو موجود، وإلا نرجع للمنفذ 8000 للتشغيل المحلي
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
