# Copyright (C) 2024 Sugar Labs, Inc.
# Copyright (C) 2025 Mebin Thattil <mail@mebin.in>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
import hmac, hashlib
import os
from dotenv import load_dotenv

router = APIRouter(tags=["webhook"])

# setup logging
logger = logging.getLogger("sugar-ai")

# Load environment variables
load_dotenv()

# Load secrets from env
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
REPO_PATH_LOCALLY = os.getenv('REPO_PATH_LOCALLY')
GIT_PATH = os.getenv('GIT_PATH')

_webhook_configured = all([WEBHOOK_SECRET, REPO_PATH_LOCALLY, GIT_PATH])

if not _webhook_configured:
    _missing = [
        name for name, val in [
            ("WEBHOOK_SECRET", WEBHOOK_SECRET),
            ("REPO_PATH_LOCALLY", REPO_PATH_LOCALLY),
            ("GIT_PATH", GIT_PATH),
        ]
        if not val
    ]
    logger.warning(
        "Webhook is disabled — missing env var(s): %s. "
        "The /webhook endpoint will return 503 until they are set.",
        ", ".join(_missing),
    )


def verify_github_signature(body: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    if not _webhook_configured:
        return False
    if not signature:
        return False
    
    try:
        sha_name, signature = signature.split('=')
        if sha_name != 'sha256':
            return False
        
        mac = hmac.new(
            WEBHOOK_SECRET.encode('utf-8'), 
            msg=body, 
            digestmod=hashlib.sha256
        )
        return hmac.compare_digest(mac.hexdigest(), signature)
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False


@router.post("/webhook")
async def webhook(request: Request):
    """GitHub webhook endpoint for handling repository updates"""
    if not _webhook_configured:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Webhook is not configured — required env vars are missing",
            },
        )

    try:
        body = await request.body()
        signature = request.headers.get('X-Hub-Signature-256')
        
        # Verify the GitHub signature
        if not verify_github_signature(body, signature):
            logger.warning("Webhook signature verification failed")
            raise HTTPException(status_code=403, detail="Signature verification failed")
        
        logger.info("Webhook signature verified successfully")
        
        # Change to repository directory
        logger.info(f"Changing directory to: {REPO_PATH_LOCALLY}")
        
        # Perform git fetch and hard reset to avoid merge conflicts
        git_fetch_command = f"cd '{REPO_PATH_LOCALLY}' && {GIT_PATH} fetch origin main"
        logger.info(f"Executing git fetch: {git_fetch_command}")
        
        fetch_result = os.system(git_fetch_command)
        if fetch_result != 0:
            logger.error(f"Git fetch failed with exit code: {fetch_result}")
            return JSONResponse(
                status_code=500, 
                content={"status": "error", "message": "Git fetch failed"}
            )
        
        logger.info("Git fetch completed successfully")
        
        # Perform hard reset to origin/CI/CD
        git_reset_command = f"cd '{REPO_PATH_LOCALLY}' && {GIT_PATH} reset --hard origin/main"
        logger.info(f"Executing git reset: {git_reset_command}")
        
        reset_result = os.system(git_reset_command)
        if reset_result != 0:
            logger.error(f"Git reset failed with exit code: {reset_result}")
            return JSONResponse(
                status_code=500, 
                content={"status": "error", "message": "Git reset failed"}
            )
        
        logger.info("Git reset completed successfully")
        
        # Restart the service
        restart_command = "sudo systemctl restart sugarai"
        logger.info(f"Executing service restart: {restart_command}")
        
        restart_result = os.system(restart_command)
        if restart_result != 0:
            logger.error(f"Service restart failed with exit code: {restart_result}")
            return JSONResponse(
                status_code=500, 
                content={"status": "error", "message": "Service restart failed"}
            )
        
        logger.info("Service restarted successfully")
        
        return JSONResponse(
            status_code=200, 
            content={
                "status": "success", 
                "message": "Repository updated and service restarted successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook processing failed: {e}")
        return JSONResponse(
            status_code=500, 
            content={"status": "error", "message": f"Internal server error: {str(e)}"}
        )
