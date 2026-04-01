**Eye Tracking and Monitor Control**

====================================

This is a 3D eye tracker that works with your webcam. This is only in the protoype stage, and I still need to optimize heavily and implement a multi-point calibration. 

To help support this software and other open-source projects, please consider subscribing to my YouTube channel:
https://www.youtube.com/@jeoresearch

Or join for $1 per month:
https://www.youtube.com/@jeoresearch/join

Recommended webcam (used in testing, ~$37):
https://amzn.to/43of401


Usage
-----

Use Python 3.10 to 3.12. The project depends on the MediaPipe Solutions API, so install from requirements.txt and keep `mediapipe==0.10.21`.

One-click launch:
- macOS / Linux shell: `./run_tracker.sh`
- Windows batch: `run_tracker.bat`
- Setup only without starting the camera: `./run_tracker.sh --setup-only` or `run_tracker.bat --setup-only`
- Choose a different camera index: `./run_tracker.sh 1` or `run_tracker.bat 1`

The launcher will automatically:
- find Python 3.10 to 3.12
- recreate `.venv` if it is missing or incompatible
- install or refresh dependencies when `requirements.txt` changes
- start `MonitorTracking.py`

Connect a webcam. By default, camera index 0 is used. You can override it with the `CAMERA_INDEX` environment variable.

macOS setup:
- `python3.12 -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `python MonitorTracking.py`

Windows setup:
- `py -3.12 -m venv .venv`
- `.venv\Scripts\activate`
- `pip install -r requirements.txt`
- `python MonitorTracking.py`

On macOS, the script uses the built-in camera by default. If the wrong camera opens, launch with `CAMERA_INDEX=1 python MonitorTracking.py` or another index.

On Windows, if the default camera is wrong, run `set CAMERA_INDEX=1 && python MonitorTracking.py` from Command Prompt or `$env:CAMERA_INDEX=1; python MonitorTracking.py` from PowerShell.

Run the tracker:
python MonitorTracking.py

Windows will open showing:
- Integrated Eye Tracking: live video with eye landmarks, gaze rays, and calibration overlays.
- Head/Eye Debug: a 3D orbit-view with the head, gaze vectors, and the calibrated virtual monitor.


Interactive controls:
-----
- c = calibrate (screen center)
- m = toggle mouse control (disabled by default)
- F7 = optional global mouse-control toggle on Windows when `keyboard` is installed
- j/l = orbit yaw left/right
- i/k = orbit pitch up/down
- [ / ] = zoom orbit view out/in
- r = reset orbit view
- x = stamp a green marker on the monitor where your gaze hits
- q = quit

Notes
-----
Make sure you look at screen center when pressing c. The debug view won't render until you do this. 
Markers (x key) allow quick tests of where the system thinks you are looking.

Troubleshooting
-----
- If gaze appears jittery, increase filter_length.
- If the wrong camera opens, use the `CAMERA_INDEX` environment variable.
- On macOS, allow camera access for VS Code or Terminal in System Settings > Privacy & Security > Camera.
- On macOS, enable Accessibility permissions as well if you want mouse control through PyAutoGUI.
- If startup fails with a MediaPipe error, recreate the virtual environment and reinstall from requirements.txt so `mediapipe==0.10.21` is used.
- For better accuracy, use consistent lighting and this webcam: https://amzn.to/43of401.
