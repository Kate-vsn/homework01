import argparse
import collections
import json
import random
import time
import urllib.error
import urllib.request

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://ml.localhost")
parser.add_argument("--rps", type=float, default=5)
parser.add_argument("--bad", type=float, default=0.0)
parser.add_argument("--data", default="src/churn/data/housing.csv")
args = parser.parse_args()

FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
df = pd.read_csv(args.data)[FEATURES]
rows = [{k: float(v) for k, v in r.items()} for r in df.to_dict("records")]

codes: collections.Counter = collections.Counter()
versions: collections.Counter = collections.Counter()
errors = 0
start = time.time()
while True:
    row = dict(random.choice(rows))
    if random.random() < args.bad:
        row["MedInc"] = -1.0
    req = urllib.request.Request(
        f"{args.url}/v1/predict",
        data=json.dumps(row).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
            code = resp.status
            versions[body.get("version", "?")] += 1
    except urllib.error.HTTPError as e:
        code = e.code
    except (urllib.error.URLError, OSError):
        code = 0
        errors += 1
    codes[code] = codes.get(code, 0) + 1
    total = sum(codes.values())
    if total % 50 == 0:
        print(
            f"{total} запросов за {time.time() - start:.0f} с, коды {dict(codes)}, "
            f"версии {dict(versions)}, недоступен {errors}",
            flush=True,
        )
    time.sleep(1 / args.rps)
