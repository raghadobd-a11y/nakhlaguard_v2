"""
NakhlaGuard: بوابة «موقع المالك».

الفكرة: الموقع العام (الرئيسية، الأدلة، طلب انضمام المزارعين) مفتوح، وكل ما عداه
(الفحص، النخيل، التنبيهات، المزارعون، المساعد، واجهات الـAPI، والـWebSocket) لا يُفتح إلا لمن
معه مفتاح جهاز صالح. المفتاح يكون داخل الباركود المطبوع على علبة الجهاز:
    https://<موقعك>/o/<مفتاح-الجهاز>
فتح الرابط يضع كوكي HttpOnly ثم يحوّل صاحب الجهاز لصفحة الفحص. جسر الأردوينو يرسل المفتاح نفسه
في الترويسة X-Nakhla-Key.

متغيرات البيئة:
    NAKHLA_DEVICE_KEYS   مفاتيح الأجهزة مفصولة بفاصلة (مفتاح لكل جهاز/علبة). كل مفتاح 16 حرفًا فأكثر.
    NAKHLA_ADMIN_KEY     مفتاح صفحة المراجعة (admin) لاعتماد المزارعين. بدونه لا تُفتح صفحة المراجعة أبدًا.
    NAKHLA_OWNER_GATE=off   لتعطيل بوابة المالك محليًا للتجربة فقط (يطبع تحذيرًا واضحًا). لا تستخدمه على Render.
                         (لا علاقة له بـ NAKHLA_GATE الذي يخص مرشّح الضجيج في predict.py)

إن لم يوجد أي مفتاح جهاز صالح، يُولَّد مفتاح عشوائي عند كل تشغيل ويُطبع في سجل الخادم، حتى لا يبقى
الموقع مفتوحًا بالخطأ (الإغلاق هو الافتراضي).
"""
import hashlib
import hmac
import os
import secrets
import time
from urllib.parse import quote

OWNER_COOKIE = "nk_owner"
ADMIN_COOKIE = "nk_admin"
MIN_KEY_LEN = 16
COOKIE_MAX_AGE = 60 * 60 * 24 * 365

# صفحات وملفات عامة (لا تحتاج مفتاحًا)
PUBLIC_EXACT = {
    "/", "/index.html", "/owner.html", "/evidence.html", "/claims.json", "/nav.js", "/health",
    "/farmers-join.html", "/farmers/apply", "/demo/agent", "/demo/sample", "/demo/analyze", "/demo/report", "/try.html", "/assistant.html", "/install.mp4", "/favicon.ico",
    "/verification_report.html", "/verification_report.json",
}
PUBLIC_EMBED = {"/locate.html"}           # open only as the 3D trunk viewer inside the public try page (?embed=1)
PUBLIC_PREFIX = ("/o/", "/a/", "/demo-wav/")           # نقطتا الدخول بالمفتاح
ADMIN_EXACT = {"/admin.html"}
ADMIN_PREFIX = ("/admin/",)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_device_keys(env=None):
    env = os.environ if env is None else env
    keys = [k.strip() for k in (env.get("NAKHLA_DEVICE_KEYS", "") or "").split(",") if k.strip()]
    good = [k for k in keys if len(k) >= MIN_KEY_LEN]
    if len(good) != len(keys):
        print("[owner_gate] WARNING: some NAKHLA_DEVICE_KEYS entries are shorter than "
              f"{MIN_KEY_LEN} characters and were ignored.", flush=True)
    return good


def _eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


class OwnerGate:
    """ASGI خالص (يغطي http و websocket). يُضاف آخر شيء ليكون الطبقة الخارجية."""

    def __init__(self, app, device_keys=None, admin_key=None, enabled=None, now=time.time):
        self.app = app
        env_off = os.environ.get("NAKHLA_OWNER_GATE", "").lower() == "off"
        self.enabled = (not env_off) if enabled is None else enabled
        keys = load_device_keys() if device_keys is None else [k for k in device_keys if len(k) >= MIN_KEY_LEN]
        if self.enabled and not keys:
            gen = secrets.token_urlsafe(24)
            keys = [gen]
            print("[owner_gate] NAKHLA_DEVICE_KEYS is not set: using a temporary key for this run only.", flush=True)
            print(f"[owner_gate] temporary owner path: /o/{gen}", flush=True)
        self.keys = keys
        self.key_hashes = [_sha(k) for k in keys]
        ak = (os.environ.get("NAKHLA_ADMIN_KEY", "") if admin_key is None else admin_key) or ""
        self.admin_key = ak if len(ak) >= MIN_KEY_LEN else ""
        self.admin_hash = _sha(self.admin_key) if self.admin_key else ""
        if ak and not self.admin_key:
            print(f"[owner_gate] WARNING: NAKHLA_ADMIN_KEY is shorter than {MIN_KEY_LEN} characters: admin stays locked.", flush=True)
        if not self.enabled:
            print("[owner_gate] WARNING: NAKHLA_OWNER_GATE=off, the owner site is OPEN. Local testing only.", flush=True)
        self._fails = {}
        self._now = now

    # ---------- مساعدات ----------
    @staticmethod
    def _headers(scope):
        return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}

    @staticmethod
    def _cookies(headers):
        out = {}
        for part in headers.get("cookie", "").split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                out[k.strip()] = v.strip()
        return out

    def _client_ip(self, scope, headers):
        xff = headers.get("x-forwarded-for", "")
        if xff:
            return xff.split(",")[-1].strip()
        c = scope.get("client")
        return c[0] if c else "unknown"

    def _locked_out(self, ip):
        now = self._now()
        hits = [t for t in self._fails.get(ip, []) if now - t < 600]
        self._fails[ip] = hits
        return len(hits) >= 10

    def _record_fail(self, ip):
        self._fails.setdefault(ip, []).append(self._now())

    def _owner_ok(self, headers, cookies):
        hk = headers.get("x-nakhla-key", "")
        if hk and any(_eq(hk, k) for k in self.keys):
            return True
        ck = cookies.get(OWNER_COOKIE, "")
        return bool(ck) and any(_eq(ck, h) for h in self.key_hashes)

    def _admin_ok(self, headers, cookies):
        if not self.admin_key:
            return False
        hk = headers.get("x-admin-key", "")
        if hk and _eq(hk, self.admin_key):
            return True
        ck = cookies.get(ADMIN_COOKIE, "")
        return bool(ck) and _eq(ck, self.admin_hash)

    @staticmethod
    def _is_public(path):
        return path in PUBLIC_EXACT or path.startswith(PUBLIC_PREFIX)

    @staticmethod
    def _is_admin(path):
        return path in ADMIN_EXACT or path.startswith(ADMIN_PREFIX)

    # ---------- الاستجابات ----------
    @staticmethod
    async def _send(send, status, headers, body=b""):
        hdrs = [(b"content-length", str(len(body)).encode())] + [(k.encode(), v.encode()) for k, v in headers]
        await send({"type": "http.response.start", "status": status, "headers": hdrs})
        await send({"type": "http.response.body", "body": body})

    def _cookie_header(self, name, value, scheme_https):
        c = f"{name}={value}; Path=/; Max-Age={COOKIE_MAX_AGE}; HttpOnly; SameSite=Lax"
        return ("set-cookie", c + ("; Secure" if scheme_https else ""))

    async def _entry(self, scope, send, headers, which):
        """/o/<key> للمالك، و/a/<key> للمراجعة."""
        path = scope["path"]
        key = path.split("/", 2)[2] if path.count("/") >= 2 else ""
        ip = self._client_ip(scope, headers)
        https = headers.get("x-forwarded-proto", scope.get("scheme", "")) in ("https", "wss")
        if self._locked_out(ip):
            return await self._send(send, 429, [("content-type", "text/plain; charset=utf-8"), ("retry-after", "600")],
                                    b"Too many attempts. Try again later.")
        if which == "owner":
            ok = bool(key) and any(_eq(key, k) for k in self.keys)
            name, val, dest, bad = OWNER_COOKIE, _sha(key), "/test.html", "/owner.html?bad=1"
        else:
            ok = bool(key) and bool(self.admin_key) and _eq(key, self.admin_key)
            name, val, dest, bad = ADMIN_COOKIE, self.admin_hash, "/admin.html", "/owner.html?bad=1"
        if not ok:
            self._record_fail(ip)
            return await self._send(send, 303, [("location", bad), ("cache-control", "no-store")])
        await self._send(send, 303, [("location", dest), self._cookie_header(name, val, https), ("cache-control", "no-store")])

    # ---------- نقطة الدخول ----------
    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket") or not self.enabled:
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        headers = self._headers(scope)
        cookies = self._cookies(headers)
        is_ws = scope["type"] == "websocket"

        if not is_ws and scope.get("method") == "OPTIONS":      # طلبات CORS التمهيدية
            return await self.app(scope, receive, send)

        if not is_ws and path.startswith("/o/"):
            return await self._entry(scope, send, headers, "owner")
        if not is_ws and path.startswith("/a/"):
            return await self._entry(scope, send, headers, "admin")

        if self._is_admin(path):
            allowed = self._admin_ok(headers, cookies)
        elif self._is_public(path):
            allowed = True
        elif path in PUBLIC_EMBED and not is_ws and "embed=1" in scope.get("query_string", b"").decode("latin-1").split("&"):
            allowed = True
        else:
            allowed = self._owner_ok(headers, cookies)

        if allowed:
            return await self.app(scope, receive, send)

        if is_ws:
            await receive()                                     # websocket.connect
            return await send({"type": "websocket.close", "code": 4401})
        accepts_html = "text/html" in headers.get("accept", "") and scope.get("method") == "GET" and path.endswith((".html", "/"))
        if accepts_html:
            return await self._send(send, 303, [("location", "/owner.html"), ("cache-control", "no-store")])
        return await self._send(send, 401, [("content-type", "application/json"), ("cache-control", "no-store")],
                                b'{"error":"owner_key_required"}')


def owner_path(key: str) -> str:
    return "/o/" + quote(key, safe="")
