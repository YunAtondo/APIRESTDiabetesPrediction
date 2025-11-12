from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
import traceback

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
    try:
        user = authenticate_user(db, login_data.correo, login_data.contrasena)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return get_Token(user)
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during login: {str(e)}"
        )