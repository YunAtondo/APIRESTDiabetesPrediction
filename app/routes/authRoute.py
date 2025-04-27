from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..schemas.loginScheme import LoginSchema
from ..core.security import Token
from ..database.database import get_db
from ..services.authService import authenticate_user, get_Token

router = APIRouter(tags=["auth"])

@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
def login(
    login_data: LoginSchema,
    db: Session = Depends(get_db),
):
    """
    Authenticate user and return access token.
    """
    user = authenticate_user(db, login_data.correo, login_data.contrasena)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return get_Token(user)