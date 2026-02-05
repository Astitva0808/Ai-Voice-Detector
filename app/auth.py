from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import API_KEY

security = HTTPBearer()

def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    if token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return token
