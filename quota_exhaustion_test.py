import json
import time
import httpx

BASE = "http://127.0.0.1:8000"
API_KEY = "user_key_2"


def main():
    # Fetch MAX_DAILY_REQUESTS from server by doing a single request to /ask and reading quota.total.
    # (We keep this script environment-agnostic.)
    with httpx.Client(timeout=120) as client:
        # Prime: request to learn quota.total
        prime = client.post(f"{BASE}/ask", params={"question": "prime"}, headers={"X-API-Key": API_KEY})
        prime_json = prime.json()
        total = (prime_json.get("quota") or {}).get("total")
        if prime.status_code != 200 or not isinstance(total, int):
            print("prime_failed", prime.status_code)
            print(json.dumps(prime_json, ensure_ascii=False))
            return 2

        # We consumed 1 request already.
        remaining = (prime_json.get("quota") or {}).get("remaining")
        print(f"prime status={prime.status_code} remaining={remaining} total={total}")

        # Send the rest of the day's quota.
        to_send = total - 1
        rows = []
        for i in range(1, to_send + 1):
            r = client.post(f"{BASE}/ask", params={"question": f"t{i}"}, headers={"X-API-Key": API_KEY})
            try:
                body = r.json()
            except Exception:
                body = {"_raw": r.text}
            rem = (body.get("quota") or {}).get("remaining") if isinstance(body, dict) else None
            detail = body.get("detail") if isinstance(body, dict) else None
            rows.append({"n": i, "status": r.status_code, "remaining": rem, "detail": detail})
            print(f"req {i}/{to_send} status={r.status_code} remaining={rem}")

        # One more should be 429
        over = client.post(f"{BASE}/ask", params={"question": "over"}, headers={"X-API-Key": API_KEY})
        over_body = over.json()
        print("over status=", over.status_code)
        print("over body=", json.dumps(over_body, ensure_ascii=False))

        # Machine-readable summary
        print("RESULT_JSON_BEGIN")
        print(json.dumps({
            "prime": {"status": prime.status_code, "quota": prime_json.get("quota")},
            "rows": rows,
            "over": {"status": over.status_code, "body": over_body},
        }, ensure_ascii=False))
        print("RESULT_JSON_END")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
