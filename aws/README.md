# AWS deployment notes: auto-sleep backend + wake Lambda

This folder holds the infrastructure pieces that let us run the Sugar-AI
backend on EC2 without paying for idle hours:

- **GitHub Pages** (served from the repo's `docs/` folder) always serves the UI.
- **EC2** hosts the FastAPI/PyTorch backend and auto-stops after 30 minutes of
  inactivity via a CloudWatch alarm.
- **Lambda (`wake_server.py`)** exposes a Function URL so the static site can
  start the instance on demand. Expected flow:
  1. Browser tries a normal API request, network error, `wake.js` shows the
     "Wake Up Server" banner.
  2. User clicks, browser POSTs to the Lambda Function URL with `X-Wake-Token`.
  3. Lambda calls `ec2:StartInstances`.
  4. Frontend polls `GET /api/health` every few seconds until it responds,
     then reloads.

Tracking issue: https://github.com/sugarlabs/sugar-ai/issues/90

---

## 1. EC2 auto-sleep (CloudWatch alarm)

1. Open **CloudWatch > Alarms > Create alarm**.
2. Pick metric `EC2 > Per-Instance Metrics > CPUUtilization` for the Sugar-AI
   instance.
3. Statistic `Average`, period `5 minutes`.
4. Condition: **Lower / < 5**, for `6` consecutive data points (= 30 min).
5. Under **Configure actions**, add an **EC2 action > Stop this instance**.
6. Name it e.g. `sugar-ai-idle-stop`.

Docs: https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/UsingAlarmActions.html

> The backend's first request after wake will be slow (model load). The
> frontend's wake banner already covers this: users see a "Starting the AI
> server..." message until `/api/health` returns 200.

## 2. Wake Lambda

### 2.1 IAM role

Create a role `sugar-ai-wake-role` with:

- Trust policy: `lambda.amazonaws.com`
- Permissions: `AWSLambdaBasicExecutionRole` **plus** this inline policy
  (scope `Resource` down to your exact instance ARN):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["ec2:StartInstances", "ec2:DescribeInstances"],
      "Resource": "arn:aws:ec2:REGION:ACCOUNT_ID:instance/i-XXXXXXXXXXXX"
    }
  ]
}
```

If AWS rejects the scoped ARN for `ec2:DescribeInstances`, set that action's
`Resource` to `"*"`; it only returns metadata.

### 2.2 Create the function

- Runtime: **Python 3.12**
- Handler: `wake_server.handler`
- Role: the one above
- Upload `wake_server.py` (no dependencies beyond boto3, which Lambda ships).

Environment variables:

| Name             | Required | Notes |
| ---------------- | -------- | ----- |
| `INSTANCE_ID`    | yes      | `i-0abc...` of the Sugar-AI EC2 instance |
| `WAKE_TOKEN`     | strongly recommended | Shared secret echoed by the frontend via `X-Wake-Token`. Any string. |
| `ALLOWED_ORIGIN` | recommended | `https://sugarlabs.github.io`, echoed in CORS headers |

### 2.3 Function URL + throttling

1. **Configuration > Function URL > Create**, AuthType **NONE**.
2. CORS: enable, allow your GH Pages origin, allow `X-Wake-Token` header,
   methods `POST, OPTIONS`.
3. **Configuration > Concurrency > Reserved concurrency**: set to `2`. A bot
   spamming the URL cannot do more than start an already-starting instance,
   and low concurrency also caps CloudWatch log spam.

Docs:
- Function URL auth: https://docs.aws.amazon.com/lambda/latest/dg/urls-auth.html
- `ec2:StartInstances`: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Stop_Start.html

### 2.4 Test

```sh
curl -X POST "$FUNCTION_URL" \
  -H "Content-Type: application/json" \
  -H "X-Wake-Token: $WAKE_TOKEN" \
  -d '{"source":"manual-test"}'
```

Expected responses:

- `202 {"state":"starting"}` : instance was stopped, now booting.
- `200 {"state":"running"}`  : already up, no-op.
- `401 {"error":"Invalid wake token"}` : token mismatch.

## 3. Wire up the frontend

In `docs/assets/js/config.js` set:

```js
window.SUGAR_AI_CONFIG = {
    API_BASE_URL: "https://sugar-ai.sugarlabs.org",
    WAKE_LAMBDA_URL: "https://<hash>.lambda-url.<region>.on.aws/",
    WAKE_TOKEN: "<same WAKE_TOKEN you set on the Lambda>",
    WAKE_POLL_INTERVAL_MS: 5000,
    WAKE_POLL_TIMEOUT_MS: 5 * 60 * 1000
};
```

The token is public (shipped in JS), so treat it as a tripwire for naive
scrapers, not a real secret. The real guards are the reserved-concurrency
cap and the fact that `StartInstances` is idempotent.

## 4. Backend `.env` additions

```
ALLOWED_ORIGINS=https://sugarlabs.github.io,http://localhost:3000
FRONTEND_URL=https://sugarlabs.github.io/sugar-ai/dashboard.html
```

`ALLOWED_ORIGINS` drives both CORS and the OAuth `frontend_redirect`
validator: only URLs whose origin is in this list will be honored when
redirecting the user back after GitHub/Google login.

## 5. Known limitations / follow-ups

- Anyone scraping the page can extract `WAKE_TOKEN`. Per-IP rate limiting via
  a tiny DynamoDB counter (see zong0728's comment on #90) is the logical
  next layer.
- First request after wake still pays the model-load cost (~30-90s depending
  on model size). The banner keeps polling until `/api/health` returns 200.
- If you would rather skip the button, call the Lambda automatically whenever
  a normal API call fails with a network error: `wake.js` already hooks into
  `SugarApi.apiFetch` for that, the manual button is just a fallback.
