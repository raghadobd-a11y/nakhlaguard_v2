# 🌴 NakhlaGuard — Early Red Palm Weevil Detection

Full working prototype (simulated-data model + web platform + Arduino path): data generation → feature extraction → training →
inference → fusion → AI recommendation → agentic AI assistant → web UI →
low-cost hardware path (Arduino + sound sensor).

**Note on structure**: every file sits in one flat folder (no subfolders) so
the whole project can be dragged straight into a GitHub repo without any
folder-upload issues.

## Quick start

**Windows, one step:** in the project folder run `.\start_local.bat` (or double-click it). It installs missing packages, trains the model if needed, asks for the Gemini key (typed hidden, never saved to a file), opens the private pages on this laptop only (`NAKHLA_OWNER_GATE=off`), starts the server and opens `http://localhost:8000/`. The window stays busy while the server runs (that is normal); Ctrl+C stops it. It stops with a message if an older server is still using port 8000. Do not use it on Render.

Manual steps:

```bash
pip install -r requirements.txt --break-system-packages

# 1) Train the model (auto-generates simulated training data the first time)
python train_model.py

# 2) Set your Gemini key (needed for recommendations + the AI assistant)
export GEMINI_API_KEY="your-key-here"      # Windows PowerShell: $env:GEMINI_API_KEY="your-key"

# 3) Run the server
python app.py
# Open: http://localhost:8000/
```

Without a Gemini key, the server still runs and returns a placeholder
recommendation instead of failing.

To test inference directly without the server:
```bash
python predict.py
```

## Project files

```
app.py                     # Main FastAPI server (all routes)
predict.py                 # Inference on a new audio file
features.py                # Mel-spectrogram-style feature extraction
train_model.py             # Trains the RandomForest severity classifier
generate_synthetic_data.py # Simulated training audio generator
fusion.py                  # Audio+image fusion + Gemini recommendation/report
agent.py                   # Agentic AI: tool-calling loop over farm data
gemini_client.py           # Shared Gemini client with retry + model fallback
acoustic_check.py          # Pulse-evidence check (abrupt, short, isolated pulses in 150-1800 Hz; air-on-microphone and overload detection); can only LOWER an alarm
noise_sounds.py            # Synthetic non-weevil sounds (air, wind on the mic, breath, fan, hum, speech, crowd, rustle, taps) and weevil look-alikes for testing
verify_noise.py            # False-alarm test of the Try-page path: python train_model.py, then python verify_noise.py
demo_audio.py              # Public try page: runs the model on 5 s windows of a clip up to 15 s, then the pulse check
demo_fallback.py           # Ready-made answers for the public assistant when the AI is slow, busy or at its limit (labelled, never an error)
demo_report.py             # Public try page: structured report (meaning, today, 4 weeks with goal + day-by-day steps + weekly check, cost of waiting, when to call an expert); ready-made fallback
make_weevil_sound.py       # Writes the 3 test sounds in demo_wavs/ (dense pulses, sparse pulses, air only)
severity_labels.py         # Arabic->English label translation for AI output
serial_bridge.py           # PC-side bridge for the Arduino Uno version
nakhla_sensor.ino           # ESP32 + WiFi version of the sensor firmware
nakhla_sensor_uno.ino       # Arduino Uno + USB serial version
index.html / try.html / assistant.html / evidence.html / owner.html / farmers-join.html   # public site (bottom tabs: Home, Farmers, Evidence, Try, Assistant, Owner)
test.html / dashboard.html / palms.html / palm.html / agent.html / farmers.html / locate.html   # owner site (needs a device key)
admin.html                 # farmer review page (needs NAKHLA_ADMIN_KEY)
owner_gate.py              # the key gate (cookie, X-Nakhla-Key header, WebSocket)
make_owner_qr.py           # makes a device key and the printable QR image
requirements.txt
```

## Deploying on Render

- Build Command: `pip install -r requirements.txt && python train_model.py`
- Start Command: `python app.py`
- Environment variable: `GEMINI_API_KEY` = your key
- Environment variable: `NAKHLA_DEVICE_KEYS` = one secret key per device box, separated by commas (16+ characters each). Make them with `python make_owner_qr.py` (see "Owner site" below).
- Environment variable: `NAKHLA_ADMIN_KEY` = a secret (16+ characters) that opens the farmer-review page. Without it the review page stays locked.

## What the design is built around

- A low-cost contact sound sensor and a microcontroller, installed by the farmer himself.
- Four severity stages (healthy, early, moderate, advanced) instead of only infested / healthy.
- Audio analysis plus an optional photo of the palm.
- After a result: a treatment plan, an estimate of the harvest impact and a live alert.
- An assistant that looks up the farm's own data before it answers.

## What the platform includes

- 4-stage severity classification (Healthy / Early / Moderate / Advanced) with a **confidence score** per result
- AI treatment report (Gemini): diagnosis, 4-week plan, approximate impact estimate
- Photo add-on: palm age estimate + preliminary visual screening (not a diagnosis)
- Palm registry (location, age, history), **QR code per palm** -> live status page (`/palm.html?id=...`)
- Live alerts (WebSocket) + backup contact shown with every alert, 30-day re-check reminder
- Farmer feedback button ("was this result correct?") stored for future retraining
- PDF export of any report (browser print), AI assistant with 5 tools, UI in English / Arabic / Urdu on every page (one language picker in the top bar)

## Arduino / sensor side

- `nakhla_sensor_uno.ino` + `serial_bridge.py` : Uno/Nano -> USB -> PC -> `/predict-full`.
  Run the server in one terminal (`python app.py`) and the bridge in another (`python serial_bridge.py COM5`).
  Close the Arduino Serial Monitor first (it holds the port).
- **Check the timing**: at 115200 baud the sketch cannot really send 4000 samples/s as text lines.
  Time a recording (R to END). If it is longer than ~5.5 s, raise the baud rate (e.g. 250000) in BOTH the sketch and `serial_bridge.py`.
- Nano clones: Board = Arduino Nano, Processor = ATmega328P (Old Bootloader) usually; CH340 driver on Windows.

## Known limitations (be transparent about these with the judges)

1. **The model is trained on simulated data.** The generator mimics periodic larval pulses (about 100-800 Hz) over background noise
   to prove the full pipeline. A public real-recordings dataset exists (**TreeVibes on Kaggle**, 8 kHz; its "infested"
   class also includes other wood-borers). Next step: evaluate the model on it, retrain, then validate on a partner farm.
   The 4 severity stages are an assumption of the simulation; public real data is binary (clean / infested).
2. **RandomForest on hand-extracted features**, not a deep network (limited data). It can be swapped later behind the same
   `predict_severity_from_array` interface.
3. **The sensor/Arduino path is not validated on a real palm trunk.** The prototype uses a sound-sensor module (air microphone);
   the planned field design uses a contact piezo disc. Signal quality through the trunk is unmeasured.
4. **Economic and harvest-impact figures in the AI report are approximate, generated by the model**, not measurements.
5. **No authentication yet**, CORS is open, and the free Render tier can wipe the SQLite database on restart.

## Future steps

- Evaluate/retrain on real recordings (TreeVibes), then a partner farm
- Contact piezo sensor + amplifier, standalone field unit (ESP32), wireless gateway, mobile app, real SMS alerts
- Authentication and permissions; multiple backup contacts

## Testing on real recordings (TreeVibes)

TreeVibes (Kaggle `potamitis/treevibes`, CC0) is ~6 GB zipped. The extracted layout seen on our machine:
`lab\lab\clean`, `lab\lab\infested` (labels in the folder names), `field\field\train`, `field\field\test`, plus `TreeVibes.xlsx`
(folder-level labels for the original folder_1 ... folder_35: AUDIO = 1 means pest sounds). Caveats: "clean" includes non-palm trees, wind and rain;
recordings come from Crete with a drill-bit / waveguide sensor, not our surface piezo.

1. Extract the zip OUTSIDE this folder (e.g. `tar -xf archive.zip -C G:\TreeVibes`). Do not push the data to GitHub.
2. `python train_model.py` (once), then `python eval_treevibes.py "G:\TreeVibes"`
   Labels come from folder / file names (infest, clean, healthy, control) or from folder_N via the xlsx. Unlabelled files are listed as examples so the rule can be adapted.
3. Expect numbers far below the simulated 93%; report them honestly. If poor, retrain a binary model on the real data,
   splitting by tree or by the provided train / test split (never by clip).

## Demo recordings

`python make_demo_wavs.py` writes `demo_wavs/1_healthy.wav ... 4_advanced.wav` (simulated, one per stage; clearest correctly classified clip out of 60 tries)
and prints the answer YOUR trained model gives for each. Say "simulated test recording" when presenting them. Every non-healthy sample must now also pass the pulse check (at least 3 pulses), so the page never shows an infested sample that its own pulse check would veto.

`python make_weevil_sound.py` writes three extra files for trying the microphone in front of judges: `weevil_dense_sim.wav` (many short pulses, should read as infested), `weevil_sparse_sim.wav` (few pulses, should read as infested), `air_noise_sim.wav` (air only, should read "no sign"). They are simulated, not a real weevil. Play them from a phone near the laptop microphone (5 to 10 cm, volume high, quiet room) and record 10 to 12 seconds; results can vary with the speaker and the room. They are also on the Try page (play and download buttons) and are served from `/demo-wav/`.

## If the report shows the fallback text instead of Gemini

Quick check: open `/health` on the site (or `http://localhost:8000/health`). `"gemini_key_set": false` means the server did not receive `GEMINI_API_KEY` (set it in the same PowerShell window before `python app.py`, or in Render > Environment). It shows only true/false, never the key.


The page prints "could not reach Gemini" and a canned recommendation when every Gemini attempt fails. `gemini_client.py` now moves to the next model after a timeout
(40 s) instead of giving up. Run `python test_gemini.py` in the same PowerShell window where `GEMINI_API_KEY` is set: it checks the key, the network, the models
available to your key and the response time of each one. The server terminal also prints a line starting with `[NakhlaGuard] Gemini call failed:` with the real reason.

**It never hangs.** One Gemini request waits at most 35 seconds in total (20 seconds per attempt), across the retries and the backup models in `gemini_client.py`. After that the page shows the ready-written fallback text instead of waiting. Change it with the environment variables `GEMINI_TOTAL_DEADLINE` and `GEMINI_REQUEST_TIMEOUT` (seconds). Every request also carries a wording rule so the AI says "palm" and never "tree".
For the exhibition: use a key made only for it, set a quota or budget limit for that key in Google AI Studio / Cloud, and keep the key only in the Render environment (never in the code or the repo).

## Training a real-data detector (TreeVibes)

`python train_treevibes.py train "G:\TreeVibes"` trains a healthy / infested detector on the real recordings with the same 44 features the website uses.
Split by tree (never by clip): test = the dataset's own field/test trees plus clean trees 9 and 35; equal weight per tree, balanced classes; the decision threshold
is chosen on out-of-fold predictions of the training trees and applied once to the held-out trees. It prints per-tree results, detection, false alarms and ROC AUC,
and saves `nakhla_binary_model.joblib`. `python train_treevibes.py predict file.wav` classifies one recording with it.
Option `--use-lab 600` adds lab recordings to training. The website itself still uses the 4-stage simulated model until this detector is integrated.

## Baseline check (poster)

`python baseline_check.py` (after `python train_model.py`) reproduces the comparison on the poster: on the same simulated data and the same 75/25 split, a one-feature
energy threshold detects infestation about as well as the Random Forest (93% vs 96% detection) but reaches only 85% on the 4-stage task, versus 93% for the Random Forest.

## Map of visible signs on the trunk (locate.html)

Open `http://localhost:8000/locate.html`. Take or choose 1 to 4 photos of the trunk (front, right, back, left) and enter the approximate trunk height.
Gemini Vision marks EXTERNAL visible signs only (holes, brown ooze, frass, damaged fibres, crown damage) on each photo. The marks are combined on one rotatable 3D trunk
with direction labels and a height ruler, and listed in a table with side and approximate height above the ground. "Print / save PDF" gives a page for an advisor.
The page warns about dark or blurry photos before upload (rough thresholds), and has a shooting guide and an inspection checklist. Backend: `locate.py` + `POST /locate-signs`.

Zoomed second look (checkbox, slower): the server crops a small square around each marked sign (up to 6 per photo) and asks Gemini a strict yes / no / uncertain question about that crop.
Signs answered "no" are faded and left out of the count; "uncertain" ones are kept and labelled. Needs Pillow (`pip install pillow`, listed in requirements.txt). Its effect has not been measured yet:
run `python eval_locate.py "C:\\path\\to\\photos" --verify` on labelled photos (folders named infested / healthy, labels from a confirmed source) and compare the false-mark rate on healthy photos with and without it.

Acoustic hotspot from the sensors (optional): in the vibration-location page (`localize.html`) the button "Show the point on the signs trunk" opens this page with the estimated point
(median of the last ~2.5 s). It can also be typed by hand (x, y in cm on the 40 x 80 cm sensor panel, panel side, panel bottom height, trunk circumference, error, status).
The point is drawn as a red marker (red rings and a filled red dot) on the 3D trunk with an uncertainty ring when an error is given, and the page names the nearest visual sign and its distance in cm.
Agreement (distance up to max(30 cm, 2 x error)) means inspection should start there; it is NOT a confirmation of infestation. The acoustic point is simulated or uncalibrated until it is measured on wood.

Limits, on purpose: it cannot see larvae inside the trunk, it does not say where the weevil is, and it never suggests a pesticide, dose or injection point.
Heights, the angle covered by each photo, the panel placement and the 3D trunk radius are approximations. The "demo" button shows labelled sample data.
`python eval_locate.py "C:\\path\\to\\photos"` reports how often signs are marked on infested photos and the false-mark rate on healthy ones; it does not measure whether a mark is in the right place
(that needs photos with hand-drawn reference boxes).

## Noise and other sounds (wind, rain, other insects, birds, engines)

The 4-stage website model was trained on simulated recordings (pink noise + pulses), so it has seen NO wind, rain, other insects, birds or machinery.
`train_treevibes.py` can add ESC-50 (github.com/karolpiczak/ESC-50, check its licence) as not-infested examples:
`python train_treevibes.py train "G:\\TreeVibes" --noise-dir "G:\\ESC-50-master"`
Categories `wind` and `insects` are held out of training by default and reported as unseen noise (change with `--noise-holdout`).

## Noise filter (so wind, rain, other insects, birds and engines do not raise alerts)

`predict.py` loads `nakhla_binary_model.joblib` if it exists next to it. When that detector says a recording does not look infested, a "moderate" or "advanced"
result is downgraded to "healthy" and no alert is raised. The filter only downgrades, it never creates an alert. It is built with `train_treevibes.py`:

1. Evaluation run (wind and insects are held out, so you see how it behaves on noise it never heard):
   `python train_treevibes.py train "G:\\TreeVibes" --noise-dir "G:\\ESC50\\ESC-50-master" --add-simulated 60`
2. Final run (all noise included in training; this is the file the website uses):
   `python train_treevibes.py train "G:\\TreeVibes" --noise-dir "G:\\ESC50\\ESC-50-master" --add-simulated 60 --noise-holdout none`
3. Restart the server. Settings without retraining (PowerShell, before `python app.py`):
   `$env:NAKHLA_GATE="off"` disables the filter; `$env:NAKHLA_GATE_THRESHOLD="0.5"` makes it more sensitive (lower = fewer suppressed alerts, more noise getting through).

Trade-off: a stricter filter also hides weak (early-stage) signals. `--add-simulated` keeps the simulated demo recordings working. Check every demo file after training.

## Live camera in the signs page, and the adult-insect sign

`locate.html` has a "Open the camera" button: it opens the device camera inside the page (no file upload), shows a dashed frame for the trunk, and "Capture and analyze" takes a frame, sends it to
`/locate-signs`, shows the marks, and moves the side selector to the next side. "Analyze automatically after capture" (on by default) also applies to photos picked from files or taken with the phone camera app.
The live camera needs a secure context: `http://localhost:8000` works on the laptop; a phone opening `http://<laptop-ip>:8000` is blocked by the browser, so the page opens the phone's own camera app instead
(same result, one extra tap). On an https deployment (for example Render) the live camera works on phones.
A new sign type, "adult_weevil" (a clearly visible adult beetle on the trunk, crown or ground), is marked with low or medium confidence unless unmistakable. Larvae inside the trunk can never be seen.

## Teaching the model what other sounds are (wind, cars, birds, ...)

1. Public sounds: `--noise-categories all` uses all 50 ESC-50 categories as "not infested" examples. `python train_treevibes.py cv ... --noise-categories all` prints which categories still fool the model.
2. Your own sensor: `python record_dataset.py COM3 "G:\\my_noise\\wind" --count 30` records clips of one sound type (one folder per type), then add `--own-noise-dir "G:\\my_noise"` to the training command.
   Noise recorded with the same sensor on a trunk is the only data that really tests "wind on a trunk vs weevil in a trunk": ESC-50 was recorded with air microphones, so a model can pass it by recognising the microphone type.
3. No dataset covers every sound. The filter says "infested" only when a recording looks like the pulses it has seen, so an unknown sound can still trigger an alert: keep the alert as a suspicion to be confirmed in the field.

## Sensors -> location -> map inside the pages the customer uses

The map and the camera are no longer a separate page for the customer. `test.html` shows a card "Trunk map" under every analysis result, and `palm.html` shows it under the palm's history;
both embed `locate.html?embed=1` (no header, no demo button, camera only). With an alert result the camera opens by itself; with a healthy result it stays closed.
Where the acoustic point comes from: the bridge (`pc/nakhla_bridge.py` in the Arduino package) sends the average level of the three channels over the last 2.5 s with every recording
(`sensor_rms`, plus `--baseline a,b,c` measured with the sensors quiet; zero by default). `POST /predict-full` estimates the source position with `sensor_location.py`
(same model and numbers as `localize.html`; the two agree on 304 random cases), returns it as `sensor_location`, includes it in the alert, and stores the latest one per palm
(`sensor_locations.json`, read by `GET /palms/{id}/sensor-location` for the palm page). A scan without sensor levels does not change the stored point.

**3D position at inspection and in alerts.** After every check (test.html) and inside every alert card (dashboard.html) the 3D trunk is shown automatically with the estimated vibration-source point (`locate.html?embed=1&view=3d`). With sensors connected the point comes from `sensor_location.py` and is labelled "live, not calibrated" or "calibrated, ± error". With NO sensor levels in the request (for example a sample recording at the exhibition) and a result other than healthy, the server adds a SIMULATED point (`demo_location()`, same palm gives the same point) marked `mode: "demo"`; the page shows a red "demonstration, not a measurement" note. It is never stored as a sensor reading. Set `$env:NAKHLA_DEMO_LOCATION="0"` to switch the simulated point off for real use. The point is the vibration source on the panel, not a confirmed larva position (claim N7 in the register).

**The device is visible in the site.** The home page has a "The device" section (sensors -> Arduino board -> bridge -> website, what each part does, prototype status). `test.html` has a "From the device" card and `dashboard.html` a status line, both fed by `GET /device/status`: the last reading that arrived with `sensor_rms`, i.e. from `nakhla_bridge.py` (kept in memory, cleared when the server restarts). The older `serial_bridge.py` does not send `sensor_rms`, so it does not light up that line; use `nakhla_bridge.py` (add `--demo` to try it without hardware).

Honest limits: simulation-tested only; without `sensor_calibration.json` ({"alpha": ..., "gains": [..3..], "error_cm": ...}) the position is marked "not calibrated, accuracy unknown";
the calibration made in the browser page is not shared with the server yet.

**Farmers network (`farmers.html`, owner site): request a licensed farmer when you are away.** The owner opens the Farmers tab (or taps "Request a farmer" on an alert card or check result, which pre-fills palm, severity and area), filters the approved farmers by the problem (inspection, treatment, removal, pruning, general care) and area, and sends a visit request to the one he chooses: his name and number, the place, what for (inspect / treat / remove / other), when (today / 3 days / week / any time) and optional palm and notes. It is stored in `visit_requests` with status `sent` and the chosen `helper_id`. The farmer's name and number then show on the owner's screen with Call and WhatsApp buttons, and under "My visit requests" (found by the owner's phone number). Farmers join only through `farmers-join.html` and the manual review described above.
Endpoints: `POST /farmers/apply` (public), `GET /admin/applications` and `POST /admin/applications/{id}/decision` (admin key), `GET /helpers` (approved farmers, no phones, filters `area` and `expertise`), `GET /helpers/count`, `POST /visit-requests` (needs `target_helper_id`), `GET /visit-requests/mine?phone=`. `POST /helpers` is closed (410). The POSTs share the per-visitor rate limit; at most 5 requests per phone per day.
Honest limits (claim N8): prototype; the license review is manual and not connected to any government platform; a phone number is the only "login" for "My visit requests"; phone numbers and license numbers are stored as plain text in `nakhla.db`; nothing guarantees an approved farmer in an area or how fast he responds, and the farmer has no inbox on the site yet (the owner contacts him directly). Approve only people whose license you checked yourself, and say it is a prototype.

## Two sites: public site and private owner site

**Public (no key):** `index.html` (what the device is, what it gives you including the AI analysis and instant alerts, how it works, an "Agentic AI" section explaining the assistant (`agent.py`, five read-only tools; the live assistant is `agent.html`, owner site only), how to place the device on the palm, prototype status and roadmap; no collapsed sections), `evidence.html` (opens with a plain-language summary; the full claim list with recompute commands is below it), `farmers-join.html` (a farmer applies), `owner.html` (explains the QR code and takes a typed key), and `try.html` (the "Try" tab, see below). Nothing here shows anyone's data. The home page opens with a "Try it yourself" section with two live boxes (both in `demo_agent.py`). **Try the device** (`POST /demo/sample`): the visitor picks one of the four ready simulated recordings in `demo_wavs/` and the local scikit-learn model classifies it; no Gemini and no key needed, and it is labelled as simulated. **Try the Agentic AI assistant** (`POST /demo/agent`): it runs the same tool-calling agent loop with Gemini, but on a built-in sample farm of 5 made-up palms, with caps so the public cannot drain your Gemini quota (8 questions per visitor per hour, 300 per day for the whole site, 300 characters per question). Settings: `NAKHLA_PUBLIC_DEMO=off` switches it off, `NAKHLA_DEMO_IP_HOURLY`, `NAKHLA_DEMO_DAILY` change the caps. It needs `GEMINI_API_KEY` like the real assistant.

**The "Try" page (`try.html`, public, a tab next to "Owner").** A visitor can upload a recording, record with the microphone (up to 15 seconds, with a stop button and a progress bar; the first 0.4 s is dropped so the click of the button is not heard), or pick one of the four ready simulated recordings or three test sounds. The browser decodes any audio format, keeps up to 15 seconds (the loudest part if the file is longer), resamples to 4000 Hz mono 16-bit and sends a small WAV (about 120 KB at most) to `POST /demo/analyze` (local model only, nothing stored, no Gemini). **Two steps** (`demo_audio.py`): (1) the model classifies each 5 s window (step 2.5 s) and the window with the most pulses wins; (2) the pulse check (`acoustic_check.py`: 150 to 1800 Hz band-pass, 200 Hz envelope, rolling 1 s floor, peaks at least 6x the floor, at least 45 ms apart, at most 80 ms wide, at least `max(2, 0.2 x seconds)` pulses, not more than 35% of the time active) can only LOWER a non-healthy verdict to "no sign" when no pulses are heard (so air, a fan or speech no longer read as infested); it never raises a result. The page draws red marks on the waveform where pulses were heard and explains a lowered verdict in words. Claim N9 states the limit: the check was tested only on synthetic distractor sounds, not on field recordings. The page then shows the four probabilities, a risk gauge, a 3D trunk (`locate.html?embed=1&view=3d`, with a SIMULATED point marked `mode: "demo"`) and a button "Write the report" (`POST /demo/report`, same per-visitor and daily caps as the public assistant; `NAKHLA_PUBLIC_DEMO=off` switches all public demos off). **The report** shows only headings, each with a button that opens its text: what to do today, a four-week plan (one heading per week with a focus line), and when to call an expert, plus an "open all" link. The AI (named "Haris" / «حارس» in the site; Gemini underneath) fills a fixed JSON; the page escapes everything. With no key, or if the AI fails or returns junk, a ready-made report is shown and labelled "ready-made version" (never presented as AI). The report is general guidance with no doses or product names. Language: Arabic, English, Urdu. The page says plainly that the model was trained on simulated recordings and has not been tested on real palms. In `owner_gate.py` the public additions are `/try.html`, `/demo/analyze`, `/demo/report`, `/demo-wav/` (only the 7 listed files), and `/locate.html` only when the query has `embed=1`; every other owner page stays locked.

**Update 17 Nov 2026: the Try page, tightened and expanded.**
(1) *Air is no longer counted as a weevil.* In our own test (`python train_model.py` then `python verify_noise.py`; 570 synthetic non-weevil sounds) the trained model alone called about 78% of them infested, and the first version of the pulse check still let 131 of 570 through (31%): air on the microphone, breath, blowing and handling noise. The check now requires a pulse to (a) rise at least 6x above the local background, (b) be at most 80 ms wide, (c) rise from 20% to its top in at most 20 ms (a gust or breath swells slowly), (d) fall back to quiet within 0.1 s on both sides (at least 3x below the peak) and stand at least 4x above the median level within 0.12 s around it (a noise burst with deep dips does not), and (e) keep its energy in 150-1800 Hz, not in the 20-120 Hz thump band. Clip-level: more than 35% of the time active (continuous sound), more than 10 pulses per second (crackle) or low-frequency rumble above 1.2x the 150-1800 Hz band (air, breath or handling on the microphone) all count as "no pulses". Result: 0 or 1 false alarm out of 570 (two different random seeds: 1/570 and 0/570), and all 240 simulated weevil clips still heard, including clips played through a phone speaker with 0.15-0.7 s room echo. The 4 stage recordings and the 3 test sounds behave as before; on simulated 5 s clips of the training kind the check keeps 100/100 healthy, 77/100 early, 100/100 moderate, 99/100 advanced (the old check: 78 early).
(2) *"Could not judge, record again."* When air or rumble dominates the recording, or the browser saw the microphone overloaded (flat-topped samples, measured before the 4 kHz copy), or the clip is silent, `/demo/analyze` returns `inconclusive: true` with a `reason` and the page shows a grey "record again" card with tips instead of "healthy" or a stage; no report or trunk is offered. `/demo/report` answers 409 for such a recording.
(3) *Stated limits.* Taps on a table or keyboard and dropped objects ARE short pulses: 90 of 120 such sounds passed. The thresholds were tuned while testing on sounds we generated, so the figures are probably better than reality; nothing was tried on a real microphone or a real palm. The page says so in a warning line under the result. Claim N9 carries the same numbers.
(4) *The report is now comprehensive.* `demo_report.py` returns: what the result means, what to do today, four weeks (each: focus, goal, 4 to 6 day-by-day actions, what to check at the end of the week, and a ready question that opens the assistant), why not to wait (words only, no invented numbers), and when to call a specialist. The ready-made version exists for 3 result groups x 3 languages; the AI version (Gemini as "Haris") fills the same JSON, and anything missing falls back to the ready-made text. Still no doses, no brands, no cure promises; confirmation and treatment stay with the specialist.
(5) *The assistant has its own tab.* `assistant.html` (public, between Try and Owner) is the same demo agent as on the home page, on the 5-palm sample farm, and it also answers general questions. Each week of the report has an "Ask Haris about this week" button that opens it with the question already typed (nothing is sent until the visitor presses Send). The owner's real assistant stays inside the private owner site (`agent.html`).
(6) *Haris does not hang or fail in front of a visitor.* `demo_agent.demo_chat` waits at most 22 s for the AI (`NAKHLA_DEMO_AI_TIMEOUT`); if the AI is slow, busy, raises an error, answers with a "⚠️" failure text, or the visitor's hourly cap / the site's daily cap is reached, the visitor gets a ready-made answer from `demo_fallback.py` instead of an error. Its first line says plainly that the AI was not available and the answer was not written by Haris. It reads the same 5-palm sample farm through the same tool functions (which palm needs attention, which are due for a re-check, details of a palm ID) and has short fixed texts for the two general chips (protecting neighbouring palms, pruning), in Arabic, English and Urdu; any other question gets the palm list and a polite note. The report on the Try page has the same protection (`AI_REPORT_TIMEOUT_S = 25` in `demo_report.py`: after that the labelled ready-made report appears). The assistant page shows "still working" after 8 s and gives up politely after 75 s. For an exhibition on shared Wi-Fi, where every visitor has the same public IP, raise `NAKHLA_DEMO_IP_HOURLY` (for example 60) in the Render environment variables for that day; the cap protects the Gemini quota, and the ready-made answers cost nothing.
(7) *Speed and staying awake.* (a) The assistant saves each real AI answer for 6 hours (`demo_agent.py`, up to 200 answers) and shows it instantly when the same question is asked again on the sample farm; a saved answer does not use the visitor's question allowance, is marked "saved" on the page, and failed or ready-made answers are never saved. (b) At start-up the server asks the five suggested questions once in the background (`NAKHLA_DEMO_PREWARM=ar` by default; `ar,en`, `all` or `off`), so even the first click is instant; it needs `GEMINI_API_KEY`, spends about 10 AI calls per start, and stops after two failures in a row. (c) The Try page report now appears at once as a ready-made version (`POST /demo/report` with `mode=ready`, no AI, no cap) and is replaced by Haris's own version when it arrives (up to 38 s); the sections the visitor opened stay open, and if the AI fails the ready-made version stays with its label. (d) `gemini_client.py`: the model order is now `gemini-3.8-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite` (Google limited access to the 2.5 models to earlier users, so `gemini-2.5-flash` was removed); set `GEMINI_MODELS` (comma-separated, in order) to change it without editing code, for example `gemini-3.5-flash-lite,gemini-3.8-flash` for the fastest answers first. Run `python test_gemini.py` to see which models your key can use. (e) Render's free web service spins down after 15 minutes without traffic and takes about a minute to wake; a paid Starter instance stays on. Not measured: real answer times (this was built without access to Gemini); time the five questions once on the deployed site.

**Owner site (needs a device key):** `test.html`, `palms.html`, `palm.html`, `dashboard.html`, `farmers.html` (choose a farmer), `agent.html`, `locate.html`, `instructions.html`, every API (`/predict-full`, `/palms`, ...) and the `/ws/alerts` WebSocket. Without a valid key a page redirects to `owner.html` and an API answers 401. The logic is in `owner_gate.py`.

- The QR code on the device box holds `https://<your-site>/o/<device-key>`. Opening it stores an HttpOnly cookie (the cookie holds a hash, not the key) and lands on `test.html`.
- Make a key and its printable QR image: `python make_owner_qr.py --base-url https://<your-site>.onrender.com` (one key per box; add all keys to `NAKHLA_DEVICE_KEYS`). Keep the keys private: not on GitHub, not in a posted photo.
- If `NAKHLA_DEVICE_KEYS` is empty the server makes a random key for that run and prints `/o/<key>` in the log, so the owner site is never left open by mistake.
- The Arduino bridge sends the same key in the `X-Nakhla-Key` header: `set NAKHLA_KEY=<key>` (or `--key <key>` for `arduino_updated/nakhla_bridge.py`) before starting it. `NAKHLA_SERVER_URL` sets the server for `serial_bridge.py`.
- For local testing only: `NAKHLA_OWNER_GATE=off` opens the owner site (not the same variable as `NAKHLA_GATE`, which is the noise filter in `predict.py`).
- All keys see the same database for now (one deployment = one farm). Separate data per device is a future step.

### Verified farmers
- A farmer applies at `farmers-join.html` with a license type and number (for example an agricultural activity license issued through the Nama platform) and chooses expertise tags: inspection, treatment, removal, pruning, general care.
- You review each application at `/a/<NAKHLA_ADMIN_KEY>` (opens `admin.html`): open the license document, match name, number and expiry, then approve, reject, or later revoke. The review is manual. The site is not connected to Absher, Nama or GOV.SA, and must not claim it is.
- Only approved farmers appear on `farmers.html`, with their expertise and no phone number. The owner picks the farmer that fits the problem and sends a visit request; the farmer's phone then appears with Call and WhatsApp buttons. The farmer has no inbox in the site (a future step); the old self-registration (`POST /helpers`) is closed (410).
- Self-registered helpers from the old version stay in the database but are unverified, so nobody sees them.

### Exhibition (24 Nov) in this mode
- Print the QR for the demo device box and open it on your own phone or laptop in front of the judges: that is the owner site. Keep the key off the poster and off screenshots you publish.
- The rate limits still apply (`NAKHLA_RATE_MAX`, `NAKHLA_RATE_GLOBAL`; audio above 6 MB refused) and protect the Gemini quota. Use a key made only for the exhibition, set a quota for it in Google AI Studio / Cloud, and delete it afterwards.
- The assistant and the palm pages read the database: register only demo palms.
- Installation video: put a file named `install.mp4` next to `app.py`; the home page then shows it instead of the placeholder.

## Evidence and verification (nothing "from my head")

- `claims.json` is the register of every number and fact: its kind (measured on simulation, measured on real public data, from a published source, or openly NOT claimed), its source with a link, and the command that recomputes it.
- `evidence.html` shows the register inside the site (linked from the home page): the claims, filters by kind, the references with links and a status for each (fully or partly checked, with the date), and the result of the last verification.
- `python verify_all.py` recomputes the measurable numbers in a temporary copy (your deployed models are never touched) and writes `verification_report.html` and `.json`
  with the environment, dataset fingerprints (SHA-256) and PASS / FAIL against the range the register expects. Add `--treevibes "G:\\TreeVibes" --esc "G:\\ESC50\\ESC-50-master"` for the real-data claims, and `--poster file.pptx` to check that the poster's numbers are in the register.
- `references.md` is the reference list for the poster and the written report, generated from `claims.json` and reviewed on 14 November 2026 against each source page: R3 and R9 are marked partial (reason written under them); every other reference was read and the numbers quoted from it were compared. R11 and R18 were removed because they could not be verified.
- The literature says the early phase is much harder than the late one (about 33% sensitivity in one study), so the 80% early recall on simulation (M4) must never be presented as a field result.

## Burst features and sample-rate test (grounded in the literature)

`--features extra` now also computes the burst rate (bursts of 7 to 199 impulses with gaps under 25 ms, as defined by Mankin et al. 2016) and the share of long trains (mostly wind).
`--sr 16000` converts recordings to 16 kHz instead of 4 kHz: if the cross-validation AUC rises clearly, the 2 kHz limit of the node is costing accuracy and a faster sensor board is a priority (analysis only: the website still works at 4 kHz).
