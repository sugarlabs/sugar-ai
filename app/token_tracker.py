"""
Token usage tracking and cost estimation utilities
"""

import logging
from typing import Optional, Dict
from transformers import AutoTokenizer
from app.database import SessionLocal, TokenUsage
from datetime import datetime, timedelta
from sqlalchemy import func

logger = logging.getLogger("sugar-ai")


class TokenTracker:
    def __init__(self, model_name: str = None, tokenizer=None):
        self.model_name = model_name

        try:
            if tokenizer is not None:
                self.tokenizer = tokenizer
            elif model_name:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            else:
                self.tokenizer = None

            logger.info("TokenTracker initialized")

        except Exception as e:
            logger.warning(f"Tokenizer init failed: {e}")
            self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string"""
        if self.tokenizer:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=True)
                return len(tokens)
            except Exception as e:
                logger.error(f"Error counting tokens: {e}")
                # Fallback: rough approximation (1 token ≈ 4 characters)
                return len(text) // 4
        else:
            # Rough approximation when tokenizer unavailable
            return len(text) // 4

    def estimate_cost(
        self, prompt_tokens: int, completion_tokens: int, model_name: str
    ) -> float:
        """
        Estimate cost in dollars based on token usage

        Pricing (as of 2024 - update these as needed):
        - GPT-2/DistilGPT2: Free (open source)
        - Llama-2-7B: ~$0.0002 per 1K tokens (hosting cost)
        - Mistral-7B: ~$0.0002 per 1K tokens
        - Larger models: ~$0.0006 per 1K tokens
        """

        # Pricing per 1K tokens (in dollars)
        pricing = {
            "gpt2": 0.0,
            "distilgpt2": 0.0,
            "llama": 0.0002,
            "mistral": 0.0002,
            "phi": 0.0001,
            "default": 0.0003,
        }

        # Determine price based on model name
        price_per_1k = pricing["default"]
        model_lower = model_name.lower()

        for key in pricing:
            if key in model_lower:
                price_per_1k = pricing[key]
                break

        total_tokens = prompt_tokens + completion_tokens
        cost = (total_tokens / 1000) * price_per_1k

        return round(cost, 6)  # Return in dollars

    def track_usage(
        self,
        api_key: str,
        user_name: str,
        endpoint: str,
        question: str,
        prompt: str,
        response: str,
        model_name: str,
        response_time: float,
    ) -> Dict[str, int]:
        """
        Track token usage and save to database

        Returns:
            Dict with token counts and cost
        """

        # Count tokens
        prompt_tokens = self.count_tokens(prompt)
        completion_tokens = self.count_tokens(response)
        total_tokens = prompt_tokens + completion_tokens

        # Estimate cost
        cost = self.estimate_cost(prompt_tokens, completion_tokens, model_name)

        # Log the usage
        logger.info(
            f"Token Usage - User: {user_name} - "
            f"Prompt: {prompt_tokens} - Completion: {completion_tokens} - "
            f"Total: {total_tokens} - Cost: ${cost:.6f}"
        )

        # Save to database
        try:
            db = SessionLocal()

            usage_record = TokenUsage(
                api_key=api_key,
                user_name=user_name,
                endpoint=endpoint,
                question=question[:500],  # Limit question length
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                model_name=model_name,
                response_time_seconds=int(response_time),
                estimated_cost_cents=int(cost * 100),  # Store as cents (integer)
            )

            db.add(usage_record)
            db.commit()
            db.refresh(usage_record)

            logger.info(f"Token usage saved to database (ID: {usage_record.id})")

            db.close()

        except Exception as e:
            logger.error(f"Error saving token usage to database: {e}")

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost": cost,
        }

    def get_user_stats(self, api_key: str, days: int = 30) -> Dict:
        """Get token usage statistics for a user"""
        try:
            db = SessionLocal()

            # Calculate date range
            since_date = datetime.utcnow() - timedelta(days=days)

            # Query database
            records = (
                db.query(TokenUsage)
                .filter(
                    TokenUsage.api_key == api_key, TokenUsage.created_at >= since_date
                )
                .all()
            )

            if not records:
                return {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "avg_tokens_per_request": 0,
                }

            total_tokens = sum(r.total_tokens for r in records)
            total_cost = (
                sum(r.estimated_cost_cents for r in records) / 100
            )  # Convert cents to dollars

            db.close()

            return {
                "total_requests": len(records),
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "avg_tokens_per_request": round(total_tokens / len(records), 1),
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Error getting user stats: {e}")
            return {}

    def get_global_stats(self, days: int = 7) -> Dict:
        """Get global token usage statistics"""
        try:
            db = SessionLocal()

            since_date = datetime.utcnow() - timedelta(days=days)

            # Get all records in date range
            records = (
                db.query(TokenUsage).filter(TokenUsage.created_at >= since_date).all()
            )

            if not records:
                return {
                    "total_requests": 0,
                    "total_tokens": 0,
                    "total_cost": 0,
                    "unique_users": 0,
                }

            total_tokens = sum(r.total_tokens for r in records)
            total_cost = sum(r.estimated_cost_cents for r in records) / 100
            unique_users = len(set(r.api_key for r in records))

            # Top users
            user_usage = {}
            for r in records:
                if r.user_name not in user_usage:
                    user_usage[r.user_name] = 0
                user_usage[r.user_name] += r.total_tokens

            top_users = sorted(user_usage.items(), key=lambda x: x[1], reverse=True)[:5]

            db.close()

            return {
                "total_requests": len(records),
                "total_tokens": total_tokens,
                "total_cost": round(total_cost, 4),
                "unique_users": unique_users,
                "avg_tokens_per_request": round(total_tokens / len(records), 1),
                "top_users": [
                    {"name": name, "tokens": tokens} for name, tokens in top_users
                ],
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Error getting global stats: {e}")
            return {}
