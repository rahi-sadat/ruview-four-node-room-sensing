# Showcase Checklist

## Before the event

- Fix the router and four ESP32 positions with tape.
- Mark five floor zones: Node 1, Node 2, Node 3, Node 4, Center.
- Power all nodes from stable USB adapters.
- Disable laptop sleep during the demo.
- Confirm Docker and RuView start after a laptop restart.
- Run `check-deployment.ps1`.
- Record one successful fallback demonstration.

## Live demonstration script

1. Show four green node indicators.
2. Leave the room: dashboard should approach **Empty room**.
3. Enter near Node 1 and remain for several seconds.
4. Move to the center.
5. Move near Node 3.
6. Explain the per-node disturbance bars.
7. Show the saved calibration consistency, but call it same-room consistency,
   not general localization accuracy.

## Accurate presentation sentence

“The system performs room-specific, device-free coarse zone classification by
learning how a person in each zone changes the CSI amplitude pattern across four
ESP32 receiver links. It does not estimate exact coordinates.”
