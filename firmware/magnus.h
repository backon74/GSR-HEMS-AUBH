// GENERATED from config.py. Same constants as logic/exports magnus_dew().
#pragma once
#include <math.h>
static const float MAGNUS_A = 17.62f;
static const float MAGNUS_B = 243.12f;
static inline float magnus_dew(float t, float rh) {
  float g = logf(rh / 100.0f) + MAGNUS_A * t / (MAGNUS_B + t);
  return MAGNUS_B * g / (MAGNUS_A - g);
}
