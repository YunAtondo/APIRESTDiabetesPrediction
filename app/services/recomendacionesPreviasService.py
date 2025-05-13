from sqlalchemy.orm import Session, joinedload
from ..models.recomendacionesPreviasModel import Recomendacion

def obtener_recomendaciones_por_usuario(db: Session, usuario_id: int) -> list[Recomendacion]:
    """
    Obtiene todas las recomendaciones de un usuario específico.
    """
    return db.query(Recomendacion)\
        .options(joinedload(Recomendacion.registro))\
        .filter(Recomendacion.id_usuario == usuario_id)\
        .all()