"""Decode the newest CDP Page.captureScreenshot payload to a PNG. Dev-only review helper."""
import base64
import glob
import json
import os
import sys

LOGS = os.path.expanduser('~/.cursor/browser-logs/cdp-response-Page.captureScreenshot-*.json')


def _data(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'data' and isinstance(v, str):
                return v
            got = _data(v)
            if got:
                return got
    if isinstance(o, list):
        for v in o:
            got = _data(v)
            if got:
                return got
    return None


newest = max(glob.glob(LOGS), key=os.path.getmtime)
with open(newest) as f:
    blob = _data(json.load(f))
out = sys.argv[1]
with open(out, 'wb') as f:
    f.write(base64.b64decode(blob))
print(out, os.path.getsize(out))
