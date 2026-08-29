import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
import urllib.request
import urllib.error

__test__ = False

BASE_URL = os.getenv("CONCURRENCY_TEST_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("CONCURRENCY_TEST_API_KEY", "user_key_1")
WORKERS = int(os.getenv("CONCURRENCY_TEST_WORKERS", "20"))


def make_request() -> Tuple[int, str]:
    url = f"{BASE_URL}/ask?question=test"
    req = urllib.request.Request(url, method="POST", headers={"X-API-Key": API_KEY})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return resp.getcode(), body
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        return exc.code, body
    except Exception as exc:
        return 0, str(exc)


def main() -> None:
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        results = list(executor.map(lambda _: make_request(), range(WORKERS)))

    success = sum(1 for code, _ in results if code == 200)
    quota_exceeded = sum(1 for code, _ in results if code == 429)
    errors = [(code, body) for code, body in results if code not in (200, 429)]

    print(json.dumps({
        "total": WORKERS,
        "success": success,
        "quota_exceeded": quota_exceeded,
        "other_errors": len(errors),
        "errors": errors[:5],
    }, indent=2))


if __name__ == "__main__":
    main()
