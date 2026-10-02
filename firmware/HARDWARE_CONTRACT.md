# Hardware contract

GENERATED from `config.py` by `evaluation/exports.py`. Do not edit by hand.

Sensor: **DHT11** (0-50 C, 20-80 %RH, +/-2.0 C, +/-5.0 %RH, integer resolution).

## Modes

| Pipeline mode | Serial token | LED (GPIO) | Fan PWM | LCD line 2 |
|---|---|---|---|---|
| normal | NORMAL | Green (27) | 40%, or 30% when dew < 16 C and temp < 40 C | NORMAL |
| pre_cool | PRE_COOL | Blue (25) | 100% | PRE-COOL |
| peak_reduce | PEAK_REDUCE | Yellow (26) | 75% | PEAK REDUCE |
| comfort_override | OVERRIDE | Red (32) | 100% | OVERRIDE |

Steady red = model-triggered override (indoor estimate). Blinking red, 250 ms = live-sensor, sensor-fail or manual override.
Fan floor 30% (a 40 mm fan stalls below ~25-30%). PWM 25 kHz via MOSFET + flyback diode. Fan = A/C stand-in.

## Timing

3 s per simulated hour, 5 s during tariff peak hours (12-18). Loop length: **86 s** per 24 h day. The laptop owns the simulated clock and sends one command per hour; the sketch does not simulate time. The ESP32 reads the DHT about once a second and streams every sample, so the dashboard is a live sensor view regardless of the hour pacing.

## Serial protocol

Laptop to ESP32, one line per simulated hour: `H14,PEAK_REDUCE,75` (optional 4th field `,MODEL` when the model triggered an override). Baud 115200.

ESP32 to laptop, about 1 line per second, JSON:

```json
{"sim_hour":14,"temp":27.8,"rh":52,"dew":16.9,"planned_mode":"peak_reduce","actual_mode":"override","override_source":"live","fan_pwm":100,"sensor_ok":true}
```

`override_source` is one of none, model, live, sensor_fail, manual.

Fallback: if no command arrives for a few seconds, the firmware runs the generated `replay_table.h`. The local override stays active and always outranks a command. Start the Python bridge before the board boots (the ESP32 resets when the port opens); close the Arduino serial monitor.

## Device-side override (local, no ML)

- Live trigger is humidity only: RH >= baseline + 15 points; baseline tracking `baseRH += 0.02*(rh - baseRH)`, frozen during the hold; hold 8 s; works at any simulated hour.
- NaN / failed read: override, fan 100%. BOOT button (GPIO0) forces the same override.
- +15 %RH is a demo-scaled stand-in for the real dew-point rule; label it on screen and slide.
- Dew point: Magnus, a = 17.62, b = 243.12 (`firmware/magnus.h`, same constants as Python).

## Temperature roles (laptop vs DHT)

- Day-ahead RF on the laptop uses **outdoor** weather from the hourly dataset; it does not run on the ESP32.
- Comfort planning uses **modelled** indoor temperature (`indoor_temp_est_c`); steady red = model override.
- On-device DHT11 is **live room** sensing for local OVERRIDE / sensor_fail only. Optional offline tau fit (`tools/log_indoor.py` + `tools/fit_tau.py --write-override`) calibrates physics, not the RF.
- Fan is an A/C stand-in: decision logic demo, not thermal physics.

## GPIO

| Signal | GPIO |
|---|---|
| LED green | 27 |
| LED blue | 25 |
| LED yellow | 26 |
| LED red | 32 |
| BOOT button | 0 |
