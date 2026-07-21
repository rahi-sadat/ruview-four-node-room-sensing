# RuView Four-Node Zone Dashboard

This is a same-room, single-person **coarse zone classifier** for four ESP32-S3
CSI nodes. It estimates:

- Empty room
- Near Node 1
- Near Node 2
- Near Node 3
- Near Node 4
- Center
- Uncertain

It does not claim exact coordinates or calibrated triangulation.

## Install location

Extract/copy this folder so the final path is:

`C:\RuViewProject\RuView\zone-dashboard`

## 1. Install once

```powershell
Set-Location C:\RuViewProject\RuView

powershell -ExecutionPolicy Bypass -File `
  .\zone-dashboard\install-zone-dashboard.ps1
```

## 2. Start RuView and connect all four nodes

```powershell
Set-Location C:\RuViewProject\RuView
docker desktop start
scripts\ruview-wireless.cmd status

Invoke-RestMethod http://localhost:3000/api/v1/nodes |
  ConvertTo-Json -Depth 8
```

Do not calibrate until all four nodes are visible.

## 3. Start this dashboard

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\zone-dashboard\run-zone-dashboard.ps1
```

Open: `http://127.0.0.1:8770`

The custom pose dashboard is attached at the same origin:

`http://127.0.0.1:8770/pose`

Start `custom-pose-dashboard\run-dashboard.ps1` in another terminal if that
page reports the pose backend as offline.

## 4. Calibration order

Keep the router and all ESP32 positions fixed.

1. **Empty room - 30 seconds**
2. **Near Node 1 - 45 seconds**
3. **Near Node 2 - 45 seconds**
4. **Near Node 3 - 45 seconds**
5. **Near Node 4 - 45 seconds**
6. **Center - 45 seconds**
7. Click **Train zone model**

For each occupied zone, stand in that zone and make small natural movements:
turn, wave one hand, stand still, and take one or two small steps. Use one person.

Calibration and model files are stored in:

- `zone-dashboard\data\calibration.json`
- `zone-dashboard\data\zone_model.joblib`

They are loaded automatically on later starts.

## 5. Showcase test

Use this order:

1. Empty room
2. Enter near Node 1
3. Move to the center
4. Move near Node 3
5. Exit the room

Wait 2-4 seconds at each position because prediction probabilities are smoothed.

## 6. Record a fallback demo

Click **Record showcase**, perform the demonstration, then click **Stop
recording**. The file is saved under `zone-dashboard\data\demo_*.jsonl`.

## 7. Record CSI training sessions

Use the **CSI recording sessions** panel for per-session RuView recordings. Each
session has its own start and stop button. The wait timer runs first; the
dashboard calls RuView's recording API only after that waiting period reaches
zero.

## 8. Health check

```powershell
powershell -ExecutionPolicy Bypass -File `
  .\zone-dashboard\check-deployment.ps1
```

## Important limits

- Room-specific: moving the router or nodes requires recalibration.
- One-person demonstration is recommended.
- "Near Node" means the learned CSI disturbance pattern associated with that
  calibration zone. It is not guaranteed physical nearest-distance measurement.
- The experimental fall-like indicator is a motion-spike heuristic and must not
  be presented as a medical safety detector.
