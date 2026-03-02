# =============================================================================
# src/utils/transactional.py
# Transactional decorator (Spring-like pattern)
# =============================================================================
"""
Decorador para manejar transacciones automáticamente
Similar a @Transactional de Spring Boot

SOPORTA DOS PATRONES:
1. Service con UnitOfWork: self.uow.commit() / self.uow.rollback()
2. Service con métodos directos: self.commit() / self.rollback()
"""
from functools import wraps
from typing import Callable
from src.utils.logger import get_logger

logger = get_logger(__name__)


def transactional(func: Callable):
    """
    Decorador para manejar transacciones automáticamente.
    - Commit automático si no hay excepciones
    - Rollback automático si hay excepciones
    
    Soporta dos patrones:
    1. Service con self.uow (legacy): llama self.uow.commit()/rollback()
    2. Service con métodos directos: llama self.commit()/rollback()
    """
    @wraps(func)
    async def wrapper(self, *args, **kwargs):
        try:
            result = await func(self, *args, **kwargs)
            # Commit automático si todo salió bien
            await _commit(self)
            return result
        except Exception as e:
            # Rollback automático si hay error
            await _rollback(self)
            logger.error(f"Transaction rolled back in {func.__name__}: {e}")
            raise
    
    return wrapper


async def _commit(service):
    """
    Commit transacción. Soporta ambos patrones:
    - service.uow.commit() (legacy)
    - service.commit() (directo)
    """
    if hasattr(service, 'uow') and service.uow is not None:
        await service.uow.commit()
    elif hasattr(service, 'commit'):
        await service.commit()
    else:
        logger.warning(f"Service {service.__class__.__name__} has no commit method or uow")


async def _rollback(service):
    """
    Rollback transacción. Soporta ambos patrones:
    - service.uow.rollback() (legacy)
    - service.rollback() (directo)
    """
    if hasattr(service, 'uow') and service.uow is not None:
        await service.uow.rollback()
    elif hasattr(service, 'rollback'):
        await service.rollback()
    else:
        logger.warning(f"Service {service.__class__.__name__} has no rollback method or uow")
