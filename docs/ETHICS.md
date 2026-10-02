# Ethics and responsible design

- **Anonymised data.** The pipeline uses one building ID (`RES_EP_001` in the synthetic sample) with no addresses or occupant data. Organizer or own-logger data must be anonymised before it enters `data/raw/` or `data/own_log.csv`.
- **Equity across tariff tiers.** Savings are reported under today's flat tariff (0.18 SAR/kWh) as well as a projected peak/off-peak tariff. Under flat pricing the load shift itself saves nothing, so lower-income households must not be sold a bill reduction that depends on a tariff that does not exist yet. Device cost (200-500 SAR assumed) is shown against both cases.
- **Comfort is never traded for savings.** Every reduction is gated on the modelled indoor temperature (limit 26 C) and the planner is bounded by a rise limit. The override always holds cooling. Reported comfort is SmartCool's own indoor excursions; baseline comfort is 100% only by assumption.
- **Fail-safe on sensor loss.** On the device, a failed or NaN sensor read forces the override (fan 100%), and the BOOT button forces it manually. The local override always outranks a laptop command, and the firmware runs the stored replay table if the laptop disappears.
- **Honest labelling.** Data is synthetic until organizer data is used; literature figures are labelled with evidence ID and type; assumptions are listed in `results/claims.md`.
