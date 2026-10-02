# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: **GSR 2026 Energy Track judges**, walking up to a demo table with limited time and no prior context. The surface has to explain what SmartCool is doing and why it matters within seconds, and be legible enough to be scored.

Secondary: **a homeowner living with the system**, glancing at it the way they would a wall thermostat. Confirmed priority is judges first, with the display ambient enough to feel like a real product rather than a test harness.

Also operates it: **the AUBH team**, watching the planner and the hardware behave while building.

## Product Purpose

SmartCool is a predictive home air-conditioning controller. It pre-cools a room before the electricity tariff peak and reduces cooling during the peak, cutting peak demand without letting the room pass a comfort limit.

Confirmed definition of success: **winning the GSR Hackathon 2026 Energy Track.**

## Positioning

The mechanism is a strict three-layer separation of temperature that a neighboring project would have to rebuild rather than copy:

1. **Outdoor weather** drives a day-ahead machine-learned A/C kWh forecast.
2. **A first-order thermal model** predicts the indoor response and enforces the comfort cap.
3. **Live DHT sensing on the device** acts only as a device-side override and optional calibration.

The savings are produced by the planner and the indoor comfort gate, not by forecaster accuracy. The pipeline measures and prints this itself: swapping the RandomForest forecast for a naive one moves the peak cut by about 0.05 percentage points on the current data.

## Operating Context

- A judge approaches a table. A laptop runs the Streamlit dashboard; beside it sits an ESP32 box with a DHT11 sensor, a fan, four LEDs and an LCD.
- The laptop owns the simulated clock and paces one simulated hour every 3 seconds, 5 seconds inside the tariff peak, so a full demo day runs in roughly 86 seconds. The ESP32 only senses and executes.
- The ESP32 reads the DHT about once per second and streams every sample. `bridge/serial_bridge.py` sends one command per simulated hour over serial and republishes each telemetry packet to `data/processed/live_telemetry.json` plus an append-only `live_history.jsonl`, which the dashboard polls.
- The demo must work with no hardware attached. A replay mode walks the planned day so the surface stays reviewable, and sensor fields stay empty rather than being filled with modelled stand-ins.
- Climate context is the Eastern Province of Saudi Arabia (Khobar, Dammam, Dhahran).
- **As of this record no hardware has been run yet** — the board is not wired. Everything hardware-facing is contract and simulation so far.

## Capabilities and Constraints

- Day-ahead A/C kWh forecaster: RandomForestRegressor, seeded, held in memory, outdoor features only, no lag shorter than 24 hours.
- Indoor physics: first-order thermal model, `a = exp(-dt/tau)`, with house profiles (tight / typical / leaky). `tau` is fittable from measured A/C-off decay curves via `tools/fit_tau.py`, which labels provenance `own_logger` when used.
- Comfort: setpoint 24 °C, hard cap 26 °C, COP 2.2, tariff peak hours 12:00–18:00.
- Device contract: serial commands shaped `H14,PEAK_REDUCE,75`. On a failed or NaN sensor read, or on the BOOT button, the device forces the override to fan 100%. A local override always outranks a laptop command, and the firmware runs the stored `replay_table.h` if the laptop disappears.
- Current results on the synthetic sample (29 days): 49.17% peak reduction, −4.52% energy closed-loop (−13.93% open-loop), 431.38 SAR/year under a projected time-of-use tariff, 80.48 SAR/year under today's flat tariff, 286.2 kg CO₂/year, 100% of hours at or below the 26 °C modelled indoor cap, maximum modelled indoor 25.679 °C.
- **Implementation facts the team did not confirm as hard rules.** These are true of the code today but were explicitly not marked binding, so ask before relaxing any of them: the forecaster takes outdoor weather only and live DHT readings are never a model feature (enforced by `FORBIDDEN_FEATURE_SUBSTR` and regression tests); the synthetic dataset and the modelled nature of indoor temperature are disclosed in the README and docs; physical constants are traceable to `docs/EVIDENCE.md`.

## Brand Commitments

- Name: **SmartCool**. Team: AUBH. Venue: GSR Hackathon 2026, Energy Track.
- **Binding:** every value shown carries its provenance — measured (from the device), modelled, or replayed. A modelled number is never presented as a sensor reading, and an absent sensor reading is shown as absent rather than substituted.
- **Binding:** no AI or Cursor credits anywhere in git history, including commit trailers and co-author lines.

## Evidence on Hand

- `docs/EVIDENCE.md` — evidence register E1–E10, each entry carrying its region, type, how it is used, and an honest status and quality flag.
- `docs/ETHICS.md` — anonymisation, tariff equity, comfort, fail-safe and labelling commitments.
- `results/kpis.json`, `results/claims.md`, `results/ablation.json`, `results/sensitivity.csv`, `results/demo_day.json`.
- `firmware/HARDWARE_CONTRACT.md`, `firmware/replay_table.h`, `firmware/magnus.h`.
- `HEMS_Sample_Dataset.xlsx` — the sample dataset, **synthetic**, 720 rows, one building ID `RES_EP_001`.

Absences that future work must not paper over:

- No public dataset pairing hourly indoor temperature with household A/C kWh for Khobar, Dammam or Dhahran homes was found (searched 2 Oct 2026). There is no measured indoor data in this project; the indoor layer is a calibrated model.
- Saudi residential electricity is flat-rate today at 0.18 SAR/kWh. Time-of-use savings figures are projected against a tariff that does not yet exist, and the flat-tariff figure must be reported alongside them.
- E1–E4 and E6–E8 are abstract-level only and have not been read in full. E5 is read in full and carries quality flags. E9 is rejected for calibration and must never be cited for A/C or Eastern Province conditions.
- Literature figures such as the 30–40% range in E7 are modelled literature results, never SmartCool's own.

## Product Principles

1. **Provenance beats polish.** A number without a known origin does not ship. Measured, modelled and replayed stay visibly distinct.
2. **Comfort is a constraint, not a tradeoff.** Every reduction is gated on modelled indoor temperature; the override always holds cooling.
3. **Fail safe toward cooling.** Sensor loss, a lost laptop, or a manual press all resolve to more cooling, never less.
4. **The planner earns the savings, not the forecaster.** Claims are attributed to the mechanism that actually produces them.
5. **Legible in seconds, calm enough to live with.** A judge must understand it at a glance; a resident must not be alarmed by it.

## Accessibility & Inclusion

Established in `docs/ETHICS.md`: savings must be reported under today's flat tariff as well as a projected peak tariff, because under flat pricing the load shift alone saves nothing. Lower-income households must not be sold a bill reduction that depends on a tariff that does not exist yet, and assumed device cost (200–500 SAR) is shown against both cases.
