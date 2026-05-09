"""Wake the Sugar-AI EC2 instance from the static GitHub Pages frontend.

Deployed behind a Lambda Function URL (AuthType: NONE). The shared
X-Wake-Token header, Lambda reserved concurrency, and the fact that
StartInstances is idempotent are what keep this public endpoint from being
abused.

Environment variables:
    INSTANCE_ID      EC2 instance id to start (i-XXXX...)
    WAKE_TOKEN       Shared token expected in the X-Wake-Token header.
                     If unset, token validation is skipped.
    ALLOWED_ORIGIN   Origin echoed back in CORS headers.

IAM: ec2:StartInstances + ec2:DescribeInstances on the instance ARN.
"""

import json
import logging
import os

import boto3
from botocore.exceptions import ClientError

log = logging.getLogger()
log.setLevel(logging.INFO)

INSTANCE_ID = os.environ.get("INSTANCE_ID", "")
WAKE_TOKEN = os.environ.get("WAKE_TOKEN", "")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

ec2 = boto3.client("ec2")


def _cors_headers():
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, X-Wake-Token",
        "Content-Type": "application/json",
    }


def _response(status, body):
    return {
        "statusCode": status,
        "headers": _cors_headers(),
        "body": json.dumps(body),
    }


def _get_header(event, name):
    headers = event.get("headers") or {}
    return headers.get(name) or headers.get(name.lower()) or ""


def handler(event, _context):
    method = (event.get("requestContext", {}).get("http", {}).get("method")
              or event.get("httpMethod")
              or "POST").upper()

    if method == "OPTIONS":
        return _response(204, {})

    if not INSTANCE_ID:
        log.error("INSTANCE_ID env var is not configured")
        return _response(500, {"error": "Server not configured"})

    if WAKE_TOKEN:
        supplied = _get_header(event, "X-Wake-Token")
        if supplied != WAKE_TOKEN:
            log.warning("wake token mismatch")
            return _response(401, {"error": "Invalid wake token"})

    try:
        desc = ec2.describe_instances(InstanceIds=[INSTANCE_ID])
        state = desc["Reservations"][0]["Instances"][0]["State"]["Name"]
    except (ClientError, IndexError, KeyError) as exc:
        log.exception("describe_instances failed")
        return _response(500, {"error": f"describe failed: {exc}"})

    if state in ("running", "pending"):
        return _response(200, {"state": state, "message": "Server is already starting or running."})

    if state != "stopped":
        return _response(409, {"state": state, "message": "Instance is in a transitional state; try again shortly."})

    try:
        ec2.start_instances(InstanceIds=[INSTANCE_ID])
    except ClientError as exc:
        log.exception("start_instances failed")
        return _response(500, {"error": f"start failed: {exc}"})

    log.info("Started EC2 instance %s", INSTANCE_ID)
    return _response(202, {"state": "starting", "message": "Server start requested."})
