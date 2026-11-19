/* NakhlaGuard: shared navigation. A top bar (brand + language button) and a bottom tab bar, the same on every page.
   Skips itself inside an iframe (the embedded trunk map) and when ?embed=1. data-mode="tabs" on the script tag shows only the tab bar. */
(function () {
  if (window.top !== window.self) return;
  if (new URLSearchParams(location.search).get("embed") === "1") return;
  var me = document.currentScript, tabsOnly = !!(me && me.dataset && me.dataset.mode === "tabs");
  var path = location.pathname.replace(/\/+$/, "") || "/";
  // موقعان: العام (رئيسية، جرّب، أدلة، تقديم المزارعين، دخول المالك) وموقع المالك الخاص (يُفتح بباركود الجهاز).
  var PUBLIC_PATHS = ["/", "/index.html", "/owner.html", "/try.html", "/assistant.html", "/evidence.html", "/farmers-join.html", "/verification_report.html"];
  var isPublic = PUBLIC_PATHS.indexOf(path) >= 0;
  var PUBLIC_TABS = [
    { href: "/index.html", match: ["/", "/index.html"], icon: "🏠", t: { en: "Home", ar: "الرئيسية", ur: "ہوم" } },
    { href: "/farmers-join.html", match: ["/farmers-join.html"], icon: "🧑‍🌾", t: { en: "Farmers", ar: "للمزارعين", ur: "کسان" } },
    { href: "/evidence.html", match: ["/evidence.html", "/verification_report.html"], icon: "📚", t: { en: "Evidence", ar: "الأدلة", ur: "ثبوت" } },
    { href: "/try.html", match: ["/try.html"], icon: "🎙️", t: { en: "Try", ar: "جرّب", ur: "آزمائیں" } },
    { href: "/assistant.html", match: ["/assistant.html"], icon: "🤖", t: { en: "Assistant", ar: "المساعد", ur: "معاون" } },
    { href: "/owner.html", match: ["/owner.html"], icon: "🔑", t: { en: "Owner", ar: "المالك", ur: "مالک" } }
  ];
  var OWNER_TABS = [
    { href: "/test.html", match: ["/test.html"], icon: "🎙️", t: { en: "Check", ar: "افحص", ur: "جانچ" } },
    { href: "/palms.html", match: ["/palms.html", "/palm.html"], icon: "🌴", t: { en: "My palms", ar: "نخيلي", ur: "میری کھجوریں" } },
    { href: "/dashboard.html", match: ["/dashboard.html"], icon: "🔔", t: { en: "Alerts", ar: "التنبيهات", ur: "الرٹس" } },
    { href: "/farmers.html", match: ["/farmers.html"], icon: "🧑‍🌾", t: { en: "Farmers", ar: "المزارعون", ur: "کسان" } },
    { href: "/agent.html", match: ["/agent.html"], icon: "🤖", t: { en: "Assistant", ar: "المساعد", ur: "معاون" } }
  ];
  var TABS = isPublic ? PUBLIC_TABS : OWNER_TABS;
  var MANY = TABS.length > 5;   // six tabs on the public site: slightly smaller labels so they fit on a phone

  var css = document.createElement("style");
  css.textContent =
    "body{padding-bottom:96px !important}" +
    ".nk-top{position:sticky;top:0;z-index:60;display:flex;align-items:center;justify-content:space-between;gap:10px;padding:10px 16px;background:#0F2F26;color:#EDE6D6;box-shadow:0 2px 10px rgba(0,0,0,.18)}" +
    ".nk-brand{color:#EDE6D6;text-decoration:none;font-size:1.05rem;letter-spacing:.2px}.nk-brand b{color:#D9A441}" +
    ".nk-top #langToggle{position:static !important;margin:0 !important;inset:auto !important;transform:none !important;background:transparent !important;color:#EDE6D6 !important;border:1px solid rgba(237,230,214,.5) !important;border-radius:999px !important;padding:6px 14px !important;font-size:.85rem !important;box-shadow:none !important;cursor:pointer}" +
    "a.back[href='/index.html']{display:none !important}" +
    ".nk-tabs{position:fixed;left:0;right:0;bottom:10px;z-index:70;margin:0 auto;width:calc(100% - 20px);max-width:560px;display:flex;background:#123A2E;border-radius:20px;padding:6px;box-shadow:0 8px 24px rgba(0,0,0,.28)}" +
    ".nk-tab{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;padding:7px 2px;border-radius:14px;color:#cfd8cf;text-decoration:none;font-size:.72rem;line-height:1.2;text-align:center}" +
    ".nk-tab span{font-size:1.25rem}.nk-tab.on{background:#1F5C4A;color:#fff;box-shadow:inset 0 -3px 0 #D9A441}" +
    (MANY ? ".nk-tab{font-size:.66rem;padding:7px 1px}.nk-tab span{font-size:1.15rem}" : "") +
    ".nk-seg{display:inline-flex;border:1px solid rgba(237,230,214,.5);border-radius:999px;overflow:hidden}" +
    ".nk-seg button{all:unset;cursor:pointer;padding:5px 11px;font-size:.78rem;line-height:1.7;color:#EDE6D6;white-space:nowrap}" +
    ".nk-seg button+button{border-inline-start:1px solid rgba(237,230,214,.3)}" +
    ".nk-seg button.on{background:#D9A441;color:#123A2E;font-weight:700}" +
    ".nk-seg button:focus-visible{outline:2px solid #fff;outline-offset:-2px}" +
    "@media print{.nk-top,.nk-tabs{display:none !important}body{padding-bottom:0 !important}}";
  document.head.appendChild(css);
  function lang() { var l = document.documentElement.lang; return l === "ar" || l === "ur" ? l : "en"; }
  if (!tabsOnly) {
    var top = document.createElement("header"); top.className = "nk-top";
    top.innerHTML = '<a class="nk-brand" href="/">🌴 <b>Nakhla</b>Guard</a><span class="nk-slot"></span>';
    document.body.insertBefore(top, document.body.firstChild);
    var lt = document.getElementById("langToggle");
    if (lt) {
      // three buttons (Arabic / English / Urdu) instead of the single cycling button; the old button stays in the page, hidden, as a fallback
      lt.style.display = "none";
      var seg = document.createElement("div"); seg.className = "nk-seg"; seg.setAttribute("role", "group"); seg.setAttribute("aria-label", "Language");
      [["ar", "العربية"], ["en", "English"], ["ur", "اردو"]].forEach(function (p) {
        var b = document.createElement("button"); b.type = "button"; b.textContent = p[1]; b.setAttribute("data-l", p[0]);
        b.addEventListener("click", function () { if (typeof window.setLang === "function") window.setLang(p[0]); });
        seg.appendChild(b);
      });
      top.querySelector(".nk-slot").appendChild(seg);
    }
  }
  var bar = document.createElement("nav"); bar.className = "nk-tabs"; bar.setAttribute("aria-label", "NakhlaGuard");
  TABS.forEach(function (t) { var a = document.createElement("a"); a.className = "nk-tab" + (t.match.indexOf(path) >= 0 ? " on" : ""); a.href = t.href; a.innerHTML = "<span>" + t.icon + "</span><i style='font-style:normal'></i>"; a._t = t.t; bar.appendChild(a); });
  document.body.appendChild(bar);
  function label() {
    var l = lang(); Array.prototype.forEach.call(bar.children, function (a) { a.lastChild.textContent = a._t[l]; });
    Array.prototype.forEach.call(document.querySelectorAll(".nk-seg button"), function (b) { b.classList.toggle("on", b.getAttribute("data-l") === l); b.setAttribute("aria-pressed", b.getAttribute("data-l") === l ? "true" : "false"); });
  }
  label(); new MutationObserver(label).observe(document.documentElement, { attributes: true, attributeFilter: ["lang"] });
})();

/* Device status line: shows whether the sensor device has sent a reading to this server (GET /device/status).
   Usage: window.nkDeviceStatus("elementId"). Memory only on the server: it resets when the server restarts. */
(function () {
  var TXT = {
    ar: { none: "الجهاز: لم تصل أي قراءة منه منذ تشغيل الخادم. (يمكنك رفع تسجيل أو تجربة تسجيل جاهز.)", some: function (a, p, c) { return "الجهاز: آخر قراءة وصلت " + a + " (النخلة " + p + (c ? "، النتيجة: " + c : "") + ")."; }, off: "الجهاز: تعذّر الاتصال بالخادم." },
    en: { none: "Device: no reading has arrived since the server started. (You can upload a recording or try a ready sample.)", some: function (a, p, c) { return "Device: last reading arrived " + a + " (palm " + p + (c ? ", result: " + c : "") + ")."; }, off: "Device: could not reach the server." },
    ur: { none: "ڈیوائس: سرور چلنے کے بعد سے اس سے کوئی ریڈنگ نہیں آئی۔ (آپ ریکارڈنگ اپ لوڈ کر سکتے ہیں یا تیار نمونہ آزمائیں۔)", some: function (a, p, c) { return "ڈیوائس: آخری ریڈنگ " + a + " آئی (کھجور " + p + (c ? "، نتیجہ: " + c : "") + ")۔"; }, off: "ڈیوائس: سرور سے رابطہ نہیں ہو سکا۔" }
  };
  var CLS = { ar: { "سليم": "سليمة", "مبكرة": "مبكرة", "متوسطة": "متوسطة", "متقدمة": "متقدمة" }, en: { "سليم": "Healthy", "مبكرة": "Early stage", "متوسطة": "Moderate", "متقدمة": "Advanced" }, ur: { "سليم": "صحت مند", "مبكرة": "ابتدائی مرحلہ", "متوسطة": "درمیانہ", "متقدمة": "شدید" } };
  function lang() { var l = document.documentElement.lang; return l === "ar" || l === "ur" ? l : "en"; }
  function ago(iso, l) {
    var s = Math.max(0, Math.round((Date.now() - Date.parse(iso + "Z")) / 1000));
    try {
      var f = new Intl.RelativeTimeFormat(l, { numeric: "auto" });
      if (s < 60) return f.format(-s, "second");
      if (s < 3600) return f.format(-Math.round(s / 60), "minute");
      if (s < 86400) return f.format(-Math.round(s / 3600), "hour");
      return f.format(-Math.round(s / 86400), "day");
    } catch (e) { return s + " s"; }
  }
  window.nkDeviceStatus = function (id) {
    var el = document.getElementById(id); if (!el) return;
    var last = null, failed = false;
    function render() {
      var l = lang(), t = TXT[l];
      el.className = "device-status " + (failed ? "off" : last ? "on" : "idle");
      el.textContent = failed ? t.off : last ? t.some(ago(last.time, l), last.palm_id, last.class ? (CLS[l][last.class] || last.class) : "") : t.none;
    }
    function poll() {
      fetch("/device/status").then(function (r) { return r.json(); }).then(function (j) { last = j && j.last ? j.last : null; failed = false; render(); }).catch(function () { failed = true; render(); });
    }
    new MutationObserver(render).observe(document.documentElement, { attributes: true, attributeFilter: ["lang"] });
    poll(); setInterval(poll, 10000);
  };
})();
