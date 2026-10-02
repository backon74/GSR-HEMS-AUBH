"""results/claims.md and the generated README results block. Every number is tied to kpis.json, an evidence ID, or 'assumption'."""
import re

import config


def _g(d, path):
    for k in path.split('.'):
        d = d[k]
    return d


def _fmt(v):
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def claim_rows(k, abl=None):
    rng = k.get('sensitivity_range', {}).get('metrics', {})
    ours = [
        ('Peak-window A/C reduction (TOU 12-18), typical profile', 'headline.peak_reduction_pct', '%'),
        ('Energy change, closed-loop (negative = saving)', 'headline.energy_change_pct_closed_loop', '%'),
        ('Energy change, open-loop (for transparency)', 'energy_change_open_loop_pct', '%'),
        ('Comfort: hours with modelled indoor <= %.1f C' % config.COMFORT_T_MAX, 'headline.comfort_pct_indoor_modelled', '%'),
        ('Max modelled indoor temperature', 'max_indoor_c', 'C'),
        ('Override hours (model, indoor gate)', 'override_hours', 'h'),
        ('Saving per home per year, projected TOU', 'cost_sar.saved_per_year_tou', 'SAR'),
        ("Saving per home per year, today's flat tariff", 'cost_sar.saved_per_year_flat', 'SAR'),
        ('Payback, projected TOU (device cost assumed)', 'payback_months.tou', 'months'),
        ('Payback, flat tariff (device cost assumed)', 'payback_months.flat', 'months'),
        ('CO2 saved per home per year', 'co2_kg.saved_per_year', 'kg'),
        ('Condensate, optimised load', 'condensate_L_day.optimized', 'L/day'),
        ('Eastern Province, 30% adoption, TOU', 'scaling.eastern_province_tou.30pct.sar_per_year_million', 'million SAR/year'),
        ('Eastern Province, 30% adoption, flat tariff', 'scaling.eastern_province_flat.30pct.sar_per_year_million', 'million SAR/year'),
        ('Eastern Province, 30% adoption, peak reduced', 'scaling.eastern_province_tou.30pct.peak_mw_reduced', 'MW'),
    ]
    rows = [(t, _fmt(_g(k, p)), u, f'kpis.json:{p}', 'simulation (synthetic sample, modelled indoor)') for t, p, u in ours]
    for m, label in [('peak_cut_pct', 'Peak reduction range across sensitivity grid'),
                     ('energy_change_pct', 'Energy change range across sensitivity grid')]:
        if m in rng:
            r = rng[m]
            rows.append((label, f"{_fmt(r['min'])} to {_fmt(r['max'])}", '%',
                         f'kpis.json:sensitivity_range.metrics.{m}', 'simulation'))
    if abl:
        rows.append(('RF forecast beats best naive baseline (test MAE)', str(abl['rf_beats_naive']), '',
                     'ablation.json:rf_beats_naive', 'simulation'))
    lit = [
        ('Up to ~100% rise in cooling energy for RH 20% to 100% (direction of effect)', 'up to 100', '%', 'E3', 'field measurement (1990, Dhahran)'),
        ('A/C share of household kWh, KFUPM housing', '73', '%', 'E4', 'field measurement (monthly)'),
        ('A/C share of household kWh, Jeddah', '66.5', '%', 'E8', 'survey / bills'),
        ('Setpoint 24 C instead of 22 C lowers consumption', '11.1', '%', 'E8', 'survey / bills'),
        ('Occupancy-based setpoints, modelled annual electricity', 'up to 38.7', '%', 'E7', 'simulation (literature range, not our result)'),
        ('Occupancy-based setpoints, modelled peak demand', 'up to 34.7', '%', 'E7', 'simulation (literature range, not our result)'),
        ('Fixed-speed split COP', '2.23-2.39', '', 'E5', 'lab measurement (Madinah, not Eastern Province)'),
        ('Recommended cooling setpoint', '23-25', 'C', 'E6', 'survey / stock model'),
    ]
    assume = [
        ('Thermal time constant tau', '5 (3 / 8)', 'h', 'assumption', 'assumed; measure with tools/fit_tau.py'),
        ('Thermal capacity C_th', '5 (3.7 / 7.3)', 'kWh_th/C', 'assumption', 'assumed, no source'),
        ('Thermostat recovery gain K_REC', str(config.K_REC), '/h', 'assumption', 'assumed'),
        ('Comfort limit', str(config.COMFORT_T_MAX), 'C', 'assumption', 'assumed; E1 suggests adaptive limits could be higher'),
        ('Device cost', f'{config.DEVICE_COST_SAR:g} ({config.DEVICE_COST_RANGE_SAR[0]:g}-{config.DEVICE_COST_RANGE_SAR[1]:g})', 'SAR', 'assumption', 'estimate'),
        ('Eastern Province residential homes', f'{config.EP_RESIDENTIAL_HOMES:,}', 'homes', 'assumption', 'UNRESOLVED; deck says 1.5M'),
        ('Heat-wave load scale', f'{config.HEATWAVE_LOAD_SCALE} (low {config.HEATWAVE_LOAD_SCALE_LOW})', 'x', 'assumption', 'upper-bound style; E5 one lab unit +30% power'),
        ('Projected peak tariff', f'{config.TARIFF_PEAK_SAR}', 'SAR/kWh', 'assumption', 'projected demand-response scenario'),
        ('Flat tariff today', f'{config.TARIFF_FLAT_SAR}', 'SAR/kWh', 'assumption', 'matches E5'),
        ('Annualisation days', str(config.ANNUAL_COOLING_DAYS), 'days', 'assumption', 'overstates: sample is July, the peak month'),
    ]
    return ours_to_sections(rows, lit, assume)


def ours_to_sections(rows, lit, assume):
    return {'Our results (results/kpis.json)': rows, 'Literature figures (not our results)': lit,
            'Assumptions': assume}


def claims_md(k, abl=None):
    L = ['# Claims register', '',
         f"Generated by `evaluation/report.py`. data_source = **{k['data_source']}**. Every number used in slides or video must appear here.",
         '']
    for title, rows in claim_rows(k, abl).items():
        L += [f'## {title}', '', '| Claim | Value | Unit | Source | Type |', '|---|---|---|---|---|']
        for r in rows:
            L.append('| ' + ' | '.join(str(x) for x in r) + ' |')
        L.append('')
    return '\n'.join(L)


START, END = '<!-- KPIS:START -->', '<!-- KPIS:END -->'


def readme_block(k):
    h = k['headline']
    rng = k.get('sensitivity_range', {}).get('metrics', {})
    pc, ec = rng.get('peak_cut_pct'), rng.get('energy_change_pct')
    L = [START,
         f"_Generated from `results/kpis.json` (data source: **{k['data_source']}**; simulation, indoor temperature modelled). Do not edit by hand._", '',
         f"- **Peak reduction** (12:00-18:59 A/C load): **{h['peak_reduction_pct']}%**"
         + (f" (range {pc['min']:g} to {pc['max']:g}% across house profiles and comfort limits)" if pc else ''),
         f"- **Energy change, closed-loop**: **{h['energy_change_pct_closed_loop']}%** (negative = saving; open-loop would be "
         f"{k['energy_change_open_loop_pct']}%)"
         + (f", range {ec['min']:g} to {ec['max']:g}%" if ec else ''),
         f"- **Comfort**: modelled indoor temperature stays at or below {config.COMFORT_T_MAX:g} C for **{h['comfort_pct_indoor_modelled']}%** of hours "
         f"(max {k['max_indoor_c']} C; baseline is 100% by construction)",
         f"- **Per home per year**: {k['cost_sar']['saved_per_year_tou']} SAR under projected TOU (0.18 / 0.30), "
         f"{k['cost_sar']['saved_per_year_flat']} SAR under today's flat 0.18 tariff",
         f"- **CO2**: {k['co2_kg']['saved_per_year']} kg per home per year; **condensate**: {k['condensate_L_day']['optimized']} L/day (formula unconfirmed)",
         '',
         'Not shown: payback, Eastern Province totals. See `results/kpis.json` (payback and adoption-scaled ranges) and `results/claims.md`. '
         'Literature figures (E1-E10) live in `docs/EVIDENCE.md`, never in this block.', END]
    return '\n'.join(L)


def update_readme(k, path='README.md'):
    txt = open(path).read()
    block = readme_block(k)
    if START in txt:
        txt = re.sub(re.escape(START) + r'.*?' + re.escape(END), lambda m: block, txt, flags=re.S)
    else:
        txt = txt.rstrip('\n') + '\n\n## Results\n' + block + '\n'
    open(path, 'w').write(txt)
