---
name: SmartCool Console
description: Aneroid weather-station instrument under a real Dammam sky
colors:
  night-void: "#05080b"
  plate: "#0d1217"
  plate-glass: "rgba(12,17,21,0.92)"
  tty-bg: "#070b0e"
  ink: "#eef3f7"
  ink-2: "#b6c4cf"
  ink-3: "#9dabb6"
  ink-void: "#46535e"
  ink-tty: "#9fb0bd"
  brass: "#c9a227"
  mode-normal: "#3f8f6b"
  mode-precool: "#4a86d8"
  mode-peak: "#d8a12a"
  mode-override: "#cf5340"
  mode-fallback: "#444"
  live: "#5fd999"
  fault: "#ff8f6b"
  stale: "#9a7a3a"
  offline: "#5d6d79"
  trace-warm: "#e0a45c"
  trace-cool: "#7fa9c9"
  trace-pale: "#b9d4c9"
  trace-model: "#6f9fe0"
  trace-clay: "#c2614a"
  bezel: "rgba(255,255,255,0.14)"
  tick: "#7f8d99"
  tick-major: "#dde5ec"
  numeral: "#cfd9e2"
  hand-minute: "#cdd7e0"
  nameplate-top: "#202830"
  nameplate-bottom: "#141a20"
  nameplate-cap: "#93a2ae"
  scroll-track: "#05080b"
  scroll-thumb: "#27313a"
  scroll-thumb-hover: "#3a4752"
  sun-glow: "rgba(255,236,190,0.42)"
  moon-glow: "rgba(207,224,255,0.26)"
  moon-glow-soft: "#cfe0ff38"
  moon-maria: "rgba(120,132,150,0.30)"
  moon-maria-soft: "rgba(120,132,150,0.22)"
  moon-mid: "#dce4ef"
  moon-limb: "#aab6c8"
  airglow: "rgba(72,104,140,0.30)"
  airglow-soft: "rgba(96,120,190,0.14)"
  shadow-plate: "rgba(0,0,0,0.42)"
  shadow-nameplate: "rgba(0,0,0,0.45)"
typography:
  display:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "1.85rem"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "1.6rem"
    fontWeight: 700
    lineHeight: 1.15
    letterSpacing: "-0.025em"
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "0.86rem"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "normal"
  label:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "0.66rem"
    fontWeight: 400
    lineHeight: 1.3
    letterSpacing: "0.17em"
  sky-label:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "0.7rem"
    fontWeight: 400
    lineHeight: 1.3
    letterSpacing: "0.16em"
  sub:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "0.72rem"
    fontWeight: 400
    lineHeight: 1.4
    letterSpacing: "0.02em"
  nameplate-read:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "1.1rem"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.02em"
  ledger-value:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "0.94rem"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "normal"
  tty:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "0.7rem"
    fontWeight: 400
    lineHeight: 1.7
    letterSpacing: "normal"
  dial-numeral:
    fontFamily: "Archivo, sans-serif"
    fontSize: "8.4px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.02em"
  readout:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "4.3rem"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.03em"
  readout-lg:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "2.3rem"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.03em"
  readout-md:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "1.5rem"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.03em"
  readout-xl-mobile:
    fontFamily: "Azeret Mono, monospace"
    fontSize: "3.2rem"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "-0.03em"
rounded:
  plate: "3px"
  nameplate: "2px"
  strip: "1px"
  scroll-thumb: "99px"
spacing:
  plate-gap: "0.7rem"
  wrap-gap: "1.7rem"
  section-pad: "3.4rem 2.4rem 0"
  wrap-pad: "9rem 2.4rem 3.6rem"
components:
  instrument-plate:
    backgroundColor: "{colors.plate-glass}"
    textColor: "{colors.ink}"
    rounded: "{rounded.plate}"
    padding: "0.9rem 1.15rem"
  instrument-plate-lead:
    backgroundColor: "{colors.plate-glass}"
    textColor: "{colors.ink}"
    rounded: "{rounded.plate}"
    padding: "1.25rem 1.4rem 1.1rem"
  dial-nameplate:
    backgroundColor: "#141a20"
    textColor: "{colors.brass}"
    rounded: "{rounded.nameplate}"
    padding: "0.4rem 0.95rem"
  section-heading:
    backgroundColor: "{colors.night-void}"
    textColor: "{colors.ink}"
    typography: "{typography.headline}"
  ledger-row:
    backgroundColor: "{colors.night-void}"
    textColor: "{colors.ink}"
    padding: "0.62rem 0.15rem"
---

# Design System: SmartCool Console

## Overview

**Creative North Star: "The Aneroid Station"**

SmartCool’s console is an instrument dial under a live Gulf sky, not a metric-card dashboard. Colour is not a fixed brand palette: it arrives from solar elevation over Dammam (`SITE_LAT` / `SITE_LON`), from `#03050b` night through civil twilight into bleached haze at high sun. Smoked graphite plates, a machined bezel, an engraved tick ring, and one brass accent carry the weather-station world.

The story is judges-first and ambient: read the room at a glance, watch the planned day step, see what the shift bought, then verify the sensors are honest. Every number keeps its provenance — measured, modelled, or replayed.

**Key Characteristics:**
- Full-bleed solar sky as the first viewport atmosphere
- 372px analog dial with nameplate under the bezel (never on the face)
- Room temperature as the dominant readout; mode/strip subordinate
- Brass as the sole accent metal; mode colours reserved for commanded state
- Monospace for measurement and provenance only

## Colors

The page has no fixed mood palette. Night and day are physics; brass and mode colours are the only committed accents.

### Primary
- **Instrument Brass** (#c9a227): Second hand, nameplate readout, SmartCool load fill, ledger emphasis. Rarity is the point.

### Secondary
- **Mode Normal** (#3f8f6b) / **Pre-cool** (#4a86d8) / **Peak** (#d8a12a) / **Override** (#cf5340): Commanded-state colour only — ribbon cells and mode titles, never decorative chrome.
- **Feed Live** (#5fd999) / **Fault** (#ff8f6b) / **Stale** (#9a7a3a) / **Offline** (#5d6d79): Sensor-feed badges.

### Neutral
- **Night Void** (#05080b): App ground under both sky bands and analytics.
- **Plate Solid** (#0d1217) / **Plate Glass** (rgba(12,17,21,0.92)): Instrument faces; 0.92 alpha so labels hold ≥4.5:1 on bleached haze as well as midnight.
- **Ink** (#eef3f7) / **Ink-2** (#b6c4cf) / **Ink-3** (#9dabb6): Primary, secondary, tertiary type.
- **Bezel Hairline** (rgba(255,255,255,0.14)): Plate borders.

### Trace accents (charts)
- Warm horizon (#e0a45c), cool zenith (#7fa9c9), pale (#b9d4c9), model (#6f9fe0), clay (#c2614a) — taken from the sky ramp so plots stay inside the committed world.

### Named Rules
**The Sky Owns Colour.** Do not invent a second brand gradient. Daylight comes from `dashboard/skyclock.py`.
**The Provenance Rule.** A number without measured / modelled / replayed does not ship.

## Typography

**Display Font:** Archivo (system-ui fallback)
**Label/Mono Font:** Azeret Mono

**Character:** Archivo carries section titles and mode names with industrial weight; Azeret Mono owns every measurement, provenance tag, sky label, and packet line.

### Hierarchy
- **Mode display** (Archivo 700, 1.85rem): Commanded mode in THE PLAN.
- **Headline** (Archivo 700, 1.6rem): Section titles below the fold.
- **Readout XL** (Azeret Mono 600, 4.3rem): Room temperature.
- **Readout LG** (Azeret Mono 600, 2.3rem): Humidity, dew, outdoor, indoor model.
- **Label** (Azeret Mono, 0.66rem, 0.17em, uppercase): Plate captions and ledger headers.

### Named Rules
**The Measurement Face Rule.** Monospace is for data and provenance, not costume technicality on body copy.

## Layout

Two stacked sky stages: NOW at ~96vh (hero), THE PLAN at ~62vh (`sc-short`). Each stage is a bottom-aligned two-column wrap — dial lane `minmax(290px, 404px)`, readout stack `minmax(320px, 700px)` — with generous top padding so the orb stays in open sky. Analytics and Sensors sit on night void with max-width 1480px and side padding 2.4rem.

Below 1100px the wrap collapses to one column, dial max-width 258px, orbit pinned to a 200px top lane so the orb never lands behind plates.

## Elevation & Depth

Hybrid: sky is atmospheric depth (gradient, haze, orb glow); plates are smoked glass with offset soft shadows and a 1px bezel highlight. Dial uses a machined radial bezel and a drop-shadow under the SVG.

### Shadow Vocabulary
- **Plate lift** (`0 14px 30px rgba(0,0,0,0.42), inset 0 1px 0 rgba(255,255,255,0.07)`): Instrument plates.
- **Nameplate** (`0 6px 16px rgba(0,0,0,0.45), inset 0 1px 0 rgba(255,255,255,0.08)`): Engraved caption under dial.
- **Dial** (`drop-shadow(0 26px 44px rgba(0,0,0,0.55))`): Bezel presence against sky.

### Named Rules
**No Halo Decoration.** Zero-offset coloured glows are reserved for the sun/moon discs only.

## Shapes

Tight industrial radii: plates 3px, nameplate 2px, strip cells 1px. Circular bezel and orb are the only round silhouettes. Hairline rules separate subordinate feed facts from the lead temperature plate.

## Components

### Instrument plate
Smoked graphite face for every now/plan readout. Lead plate pads larger for room temperature. Labels uppercase mono; values tabular mono; notes carry provenance.

### Dial + nameplate
SVG tick ring with cut-shadow engraving. Hands computed in Python each second (live) or stepped per simulated hour. Caption and digital readout live only on the nameplate under the bezel — never inside the numeral ring.

### Sky stage
Full-bleed solar layer from `sky()` / `sky_layer()`: gradient, stars (night), sun or anti-solar moon, horizon haze, site label with elevation.

### Mode strip
24 hairline cells coloured by mode; current hour outlined white. Thin variant under the feed plate so it stays subordinate.

### Ledger
Key/value rows for 29-day KPIs and session min/mean/max. Mono values, muted keys, hairline dividers.

### Raw packet tty
Azeret Mono on `#070b0e` with quiet border — the honesty surface for the live feed.

## Do's and Don'ts

### Do:
- **Do** label every value measured, modelled, or replayed.
- **Do** derive sky colour from real solar elevation over the site.
- **Do** keep room temperature the hero of the first viewport; park commanded mode under a hairline.
- **Do** keep dial captions under the bezel so hands never strike type.
- **Do** report flat-tariff savings beside time-of-use figures.

### Don't:
- **Don't** fill empty sensor tiles with modelled stand-ins.
- **Don't** put AI or Cursor credits in git history.
- **Don't** invent a second accent palette beyond brass + mode/feed states.
- **Don't** treat the surface as a generic dark dashboard of equal cards.
- **Don't** print labels inside the dial face.
