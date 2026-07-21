# RuView Custom Pose Dashboard v2

This revision fixes two live-inference mismatches:

1. It detects the modal node order from the original training recording
   `data\recordings\rec_1783755595.jsonl` and reorders current live nodes to match it.
2. It uses non-overlapping groups of 20 flattened CSI frames, matching
   `align-ground-truth.js`, instead of a sliding window.

## Replace/install

Extract the ZIP into `C:\RuViewProject\RuView` and allow it to replace the existing
`custom-pose-dashboard` folder.

## Live run

```powershell
Set-Location C:\RuViewProject\RuView
powershell -ExecutionPolicy Bypass -File `
  .\custom-pose-dashboard\run-dashboard.ps1 -Mode live
```

Open `http://127.0.0.1:8766`.

Check diagnostics:

```powershell
Invoke-RestMethod http://127.0.0.1:8766/health | ConvertTo-Json -Depth 5
```

The output should show four IDs in both `training_node_order` and
`live_node_order`, with `accepted_updates` increasing and relatively few
`rejected_updates`.

This is still a same-room, uncalibrated pilot. Stable preprocessing does not
prove that the checkpoint learned motion rather than a mean-pose shortcut.
