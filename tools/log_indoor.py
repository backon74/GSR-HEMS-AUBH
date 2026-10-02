"""Log an indoor DHT11 and an outdoor reading every 5 min to CSV (P1 own-data kit). Run >= 7 days in an
air-conditioned room and note A/C on/off with the --ac flag or by editing the ac_state column afterwards.

    python tools/log_indoor.py --indoor-port /dev/cu.usbserial-0001 --out data/own_log.csv
The indoor board prints one JSON line per second: {"temp":26,"rh":48}. Outdoor comes from a second sensor
({"temp":..}) on --outdoor-port, or is left blank for you to fill from a weather feed.
Columns: timestamp, indoor_temp_c, indoor_rh_pct, outdoor_temp_c, ac_state (data_source = own_logger on load).
"""
import argparse
import csv
import datetime as dt
import json
import os
import time


def read_json_line(ser):
    try:
        return json.loads(ser.readline().decode(errors='ignore').strip())
    except ValueError:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indoor-port', required=True)
    ap.add_argument('--outdoor-port')
    ap.add_argument('--out', default='data/own_log.csv')
    ap.add_argument('--interval-s', type=int, default=300)
    ap.add_argument('--ac', type=int, default=1, help='A/C state to record (1 on, 0 off); change between runs')
    a = ap.parse_args()
    import serial
    ind = serial.Serial(a.indoor_port, 115200, timeout=2)
    out = serial.Serial(a.outdoor_port, 115200, timeout=2) if a.outdoor_port else None
    new = not os.path.exists(a.out)
    with open(a.out, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['timestamp', 'indoor_temp_c', 'indoor_rh_pct', 'outdoor_temp_c', 'ac_state'])
        while True:
            i = read_json_line(ind)
            o = read_json_line(out) if out else {}
            w.writerow([dt.datetime.now().isoformat(timespec='seconds'), i.get('temp', ''), i.get('rh', ''),
                        o.get('temp', ''), a.ac])
            f.flush()
            time.sleep(a.interval_s)


if __name__ == '__main__':
    main()
