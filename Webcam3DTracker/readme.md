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

Headless stream calibration with 9-point regression:
- Use [StreamGazePointTracker.py](StreamGazePointTracker.py) for grayscale frame streams.
- Call `process_frame(..., calibrate_eyes=True)` once while the user looks at screen center.
- Show the 9 targets from `tracker.default_nine_point_targets()` one by one.
- After each target settles, collect 15 to 30 stable frames and call `tracker.add_calibration_sample(target_point, result)`.
- When all 9 targets are sampled, call `tracker.fit_regression_calibration()`.
- After fitting, `process_frame()` automatically returns regression-mapped screen points.

Interactive 9-point calibration in MonitorTracking.py:
- Press `c` once to lock both eye spheres while looking at screen center.
- Press `9` to open a fullscreen 3x3 target sequence on the active monitor.
- Follow each target and hold your gaze steady. The tracker auto-captures after it sees enough stable frames.
- Press `Esc` to cancel the current 9-point run without quitting the tracker.
- After all 9 targets are captured, the tracker fits a quadratic regression, saves it to `monitor_tracking_regression.npz`, and automatically switches screen output to the regression map.
- On the next launch, the tracker auto-loads the saved regression file if it exists.
- Re-running the 9-point flow overwrites the saved regression file with the new fit.

Windows will open showing:
- Integrated Eye Tracking: live video with eye landmarks, gaze rays, and calibration overlays.
- Head/Eye Debug: a 3D orbit-view with the head, gaze vectors, and the calibrated virtual monitor.


Interactive controls:
-----
- c = lock both eye spheres while looking at screen center
- s = single-point center calibration for the legacy linear mapping
- 9 = start or restart fullscreen 9-point regression calibration
- Esc = cancel the current 9-point calibration run
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
The 9-point flow depends on the eye-sphere lock from c. After you press c, the tracker will automatically restore the saved regression file if one is available.
Markers (x key) allow quick tests of where the system thinks you are looking.

Troubleshooting
-----
- If gaze appears jittery, increase filter_length.
- If the wrong camera opens, use the `CAMERA_INDEX` environment variable.
- On macOS, allow camera access for VS Code or Terminal in System Settings > Privacy & Security > Camera.
- On macOS, enable Accessibility permissions as well if you want mouse control through PyAutoGUI.
- If startup fails with a MediaPipe error, recreate the virtual environment and reinstall from requirements.txt so `mediapipe==0.10.21` is used.
- If the saved 9-point calibration becomes inaccurate, rerun the 9-point flow to overwrite it, or delete `monitor_tracking_regression.npz` and recalibrate from scratch.
- For better accuracy, use consistent lighting and this webcam: https://amzn.to/43of401.
