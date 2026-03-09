#!/usr/bin/env python3
"""Quick test to create user and test multi-turn routing."""

import asyncio
import json
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.database import get_session
from app.models import User
from app.auth import hash_password, create_access_token
from sqlalchemy import select


async def create_test_user_and_token():
    """Create test user if not exists, return JWT token."""
    async for db in get_session():
        try:
            # Check if test user exists
            result = await db.execute(select(User).where(User.username == "testuser"))
            user = result.scalars().first()
            
            if not user:
                # Create test user
                user = User(
                    username="testuser",
                    password_hash=hash_password("testpass123"),
                )
                db.add(user)
                await db.commit()
                print(f"Created test user: {user.username}")
            else:
                print(f"Using existing test user: {user.username}")
            
            # Generate token
            token = create_access_token(data={"sub": user.username})
            print(f"Bearer token: {token}")
            return token
            
        except Exception as e:
            print(f"Error: {e}")
            return None


if __name__ == "__main__":
    token = asyncio.run(create_test_user_and_token())