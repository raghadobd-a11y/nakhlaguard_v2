"""
Record many labelled clips with YOUR sensor, to teach the model what wind, cars, birds, a quiet palm... sound like on your device.

    python record_dataset.py COM3 "G:\\my_noise\\wind"  --count 30
    python record_dataset.py COM3 "G:\\my_noise\\car"   --count 30
    python record_dataset.py COM3 "G:\\my_noise\\quiet" --count 30

The last folder name is the sound type. One folder per type under the same parent (here G:\\my_noise).
Each clip is 5 seconds (set by the Arduino firmware). Clips are saved as <type>_<date>_<time>_<n>.wav and never overwrite each other.
Then train with:  python train_treevibes.py train "G:\\TreeVibes" --noise-dir "G:\\ESC50\\ESC-50-master" --own-noise-dir "G:\\my_noise" --add-simulated 60 --noise-holdout none

How to record well (this matters more than the amount):
  - fix the sensor on the trunk (or a wooden board) the same way you will use it in the field;
  - record each type at different times and places, 30 or more clips per type, and wait a few seconds between clips;
  - write down the conditions (windy afternoon, car passing, birds at dawn) in a notes file next to the folder;
  - record only sounds that are NOT weevil activity. Never put a clip from a suspected infested palm into these folders.
"""
import os
import sys
import time
import argparse
from datetime import datetime

import serial
from serial_bridge import read_one_recording, fix_rate, samples_to_wav, BAUD, SAMPLE_RATE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", help="serial port, e.g. COM3")
    ap.add_argument("folder", help="output folder; its name is the sound type, e.g. G:\\my_noise\\wind")
    ap.add_argument("--count", type=int, default=30, help="how many clips to record")
    ap.add_argument("--gap", type=float, default=2.0, help="seconds to wait between clips")
    a = ap.parse_args()
    label = os.path.basename(os.path.normpath(a.folder)) or "noise"
    os.makedirs(a.folder, exist_ok=True)
    ser = serial.Serial(a.port, BAUD, timeout=2)
    time.sleep(2)                                    # the Arduino restarts when the port opens
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0
    print(f"تسجيل {a.count} مقطع بعنوان '{label}' إلى {a.folder}. اضغط Ctrl+C للإيقاف.")
    try:
        for n in range(1, a.count + 1):
            ser.reset_input_buffer()
            ser.write(b"R")
            samples, elapsed = read_one_recording(ser)
            samples = fix_rate(samples, elapsed)
            if len(samples) < SAMPLE_RATE:
                print(f"[{n}/{a.count}] تسجيل قصير جدًا، تجاهلته.")
                continue
            path = os.path.join(a.folder, f"{label}_{stamp}_{n:03d}.wav")
            samples_to_wav(samples, path)
            saved += 1
            print(f"[{n}/{a.count}] حُفظ: {os.path.basename(path)}")
            time.sleep(a.gap)
    except KeyboardInterrupt:
        print("\nتم الإيقاف.")
    print(f"\nالمجموع المحفوظ: {saved} مقطع بالمجلد {a.folder}")


if __name__ == "__main__":
    main()
