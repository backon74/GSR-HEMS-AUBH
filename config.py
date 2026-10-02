"""Single source of truth for every constant. Other modules import from here.

Evidence IDs (E1-E10) refer to docs/EVIDENCE.md. Status tags: supported / assumed / derived / design / from_data.

Temperature / sensor roles (one codebase, three layers — do not conflate):
1. ML forecaster (logic/forecast.py): OUTDOOR weather columns only → predicts A/C kWh.
2. Indoor physics (logic/indoor_model.py): modelled T_in gates comfort in planner/engine.
3. Hardware DHT11 on ESP32: live room RH/temp for local OVERRIDE and optional tau fit via
   tools/log_indoor.py + tools/fit_tau.py. Live DHT is never an RF training feature.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Data / provenance ────────────────────────────────────────────────────────
DATA_FILE = "HEMS_Sample_Dataset.xlsx"
RAW_DIR = "data/raw"
DATA_SOURCE = "synthetic_sample"          # synthetic_sample | organizer | own_logger
OUTDOOR_PHASE_SHIFT_H = 12                # D1: rotate temp/RH/dew within each day. 0 for organizer data.
SAMPLE_TZ_NOTE = "hourly, July 2025, one building"
# Columns required after rename (outdoor climate + load). Indoor is not in the hourly sample.
REQUIRED_HOURLY_COLS = ("timestamp", "temp", "humidity", "dew_point", "solar", "ac_kwh")
# Optional override written by tools/fit_tau.py --write-override (own DHT calibration).
PROFILE_OVERRIDE_PATH = os.path.join("data", "processed", "house_profile_override.json")

# ── Tariff window / tariffs ──────────────────────────────────────────────────
PEAK_HOURS = tuple(range(12, 19))         # TOU peak 12:00-18:59 (7 h)
TARIFF_OFFPEAK_SAR = 0.18                 # also today's flat rate (E5)
TARIFF_PEAK_SAR = 0.30                    # projected DR scenario
TARIFF_FLAT_SAR = 0.18

# ── Control (legacy defaults kept) ───────────────────────────────────────────
PEAK_REDUCE_FRACTION = 0.25
PRE_COOL_BOOST_KWH = 0.15
MIN_AC_KWH = 0.10
AC_MAX_KWH = 3.6                          # max observed hourly A/C kWh in the sample
DEW_POINT_UNCOMFORTABLE = 24.0            # informational flag only (D2)
TEMP_EXTREME = 47.0                       # informational flag only (D2)
FAN_DRY_DEW_C = 16.0
FAN_DRY_TEMP_C = 40.0

# ── Indoor model (7.3) ───────────────────────────────────────────────────────
T_SET = 24.0                              # SEEC 23-25 (E6); lab setpoint (E5)
COMFORT_T_MAX = 26.0                      # assumed; Dammam homes accepted higher (E1)
PLAN_RISE_MAX_C = 1.5
COP = 2.2                                 # measured 2.23-2.39 lab (E5); field est. 2.0-2.2
DT_H = 1.0
K_REC = 0.5                               # assumed thermostat recovery gain, per hour
RECOVERY_EPS_C = 0.05
HOUSE_PROFILES = {                        # tau [h] assumed; g = COP / C_th [degC per kWh_e]
    "tight":   {"tau": 8.0, "g": 0.30, "c_th": 7.3},
    "typical": {"tau": 5.0, "g": 0.45, "c_th": 5.0},
    "leaky":   {"tau": 3.0, "g": 0.60, "c_th": 3.7},
}
DEFAULT_PROFILE = "typical"
# When PROFILE_OVERRIDE_PATH exists (from DHT log + fit_tau), indoor_model prefers its tau/g
# and provenance becomes own_logger / "own-measured". Does not affect RF FEATURES.

# ── Planner grid (7.4) ───────────────────────────────────────────────────────
PLAN_CUT_GRID = tuple(round(0.05 * i, 2) for i in range(0, 13))      # 0..0.60
PLAN_BOOST_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)                   # kWh/h
PLAN_NPRE_GRID = (1, 2, 3, 4)
PLAN_CUT_CEILING = 0.60
PLANNER = "grid"                          # grid | fixed
FIXED_PRE_COOL_HOURS = 2

# ── Forecaster / ablation (7.6) ──────────────────────────────────────────────
TEST_DAYS = 8
RANDOM_SEED = 42
DETECT_FRACTION = 0.80

# ── Scenarios ────────────────────────────────────────────────────────────────
HEATWAVE_HOURS = (13, 14, 15, 16, 17)
HEATWAVE_LOAD_SCALE = 1.25                # assumed upper; one lab unit +30% power (E5)
HEATWAVE_LOAD_SCALE_LOW = 1.10
SCENARIOS = {
    "heatwave": {"name": "heatwave", "load_scale": HEATWAVE_LOAD_SCALE, "hours": list(HEATWAVE_HOURS)},
    "heatwave_low": {"name": "heatwave", "load_scale": HEATWAVE_LOAD_SCALE_LOW, "hours": list(HEATWAVE_HOURS)},
}

# ── Sensitivity grid ─────────────────────────────────────────────────────────
SENS_RISE = (1.0, 1.5, 2.0)
SENS_KREC = (0.25, 0.5, 1.0)

# ── Economics ────────────────────────────────────────────────────────────────
DEVICE_COST_SAR = 200.0                   # assumed; range below (E5 smart-thermostat price range, estimate)
DEVICE_COST_RANGE_SAR = (200.0, 500.0)
ANNUAL_COOLING_DAYS = 365                 # assumption: overstates; sample is July (peak month)
EP_RESIDENTIAL_HOMES = 850_000            # UNRESOLVED: deck says 1.5M. Needs citation.
AVG_HOMES_PER_HOOD = 500
ADOPTION = (0.10, 0.30, 0.50)
CO2_KG_PER_KWH = 0.64                     # IEA 2023, per original code comment

# ── Hardware contract (8) ────────────────────────────────────────────────────
MODES = ("normal", "pre_cool", "peak_reduce", "comfort_override")
MODE_TOKEN = {"normal": "NORMAL", "pre_cool": "PRE_COOL", "peak_reduce": "PEAK_REDUCE",
              "comfort_override": "OVERRIDE"}
MODE_LCD = {"normal": "NORMAL", "pre_cool": "PRE-COOL", "peak_reduce": "PEAK REDUCE",
            "comfort_override": "OVERRIDE"}
MODE_LED = {"normal": "green", "pre_cool": "blue", "peak_reduce": "yellow", "comfort_override": "red"}
LED_GPIO = {"green": 27, "blue": 25, "yellow": 26, "red": 32}
FAN_PWM = {"normal": 40, "pre_cool": 100, "peak_reduce": 75, "comfort_override": 100}
FAN_PWM_NORMAL_DRY = 30
FAN_FLOOR_PCT = 30
FAN_PWM_KHZ = 25
GPIO_BOOT = 0
SENSOR = {"model": "DHT11", "temp_range_c": (0, 50), "rh_range_pct": (20, 80),
          "temp_acc_c": 2.0, "rh_acc_pct": 5.0}
OVERRIDE_SOURCES = ("none", "model", "live", "sensor_fail", "manual")
OVERRIDE_BLINK_MS = 250
LIVE_RH_DELTA = 15                        # demo-scaled stand-in for a dew-point rule
LIVE_RH_BASE_ALPHA = 0.02
LIVE_HOLD_S = 8
SECONDS_PER_SIM_HOUR = 3
SECONDS_PER_PEAK_HOUR = 5
MAGNUS_A = 17.62
MAGNUS_B = 243.12
SERIAL_BAUD = 115200
