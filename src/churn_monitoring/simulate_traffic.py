"""Send a virtual month of baseline or gradually drifting records to Vertex AI."""

from __future__ import annotations

import argparse
import json
import time

from google.cloud import aiplatform

from churn_monitoring.data import generate_customers
from churn_monitoring.schema import FEATURES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument("--phase", choices=("baseline", "drift"), required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--instances-per-day", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()
    if args.days < 2 or args.instances_per_day < 1 or args.sleep_seconds < 0:
        parser.error("days must be >= 2, instances-per-day >= 1, and sleep-seconds >= 0")

    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(args.endpoint_id)
    for day in range(args.days):
        if args.phase == "baseline":
            strength = 0.0
        else:
            strength = day / (args.days - 1)
        frame = generate_customers(
            args.instances_per_day,
            seed=args.seed + day,
            drift_strength=strength,
            include_target=False,
        )
        instances = json.loads(frame[FEATURES].to_json(orient="records"))
        response = endpoint.predict(instances=instances)
        print(
            f"virtual_day={day + 1:02d}/{args.days} phase={args.phase} "
            f"drift_strength={strength:.2f} predictions={len(response.predictions)}"
        )
        if args.sleep_seconds and day < args.days - 1:
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()

