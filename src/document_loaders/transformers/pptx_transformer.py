# =============================================================================
# src/document_loaders/transformers/pptx_transformer.py
# =============================================================================
"""
Transformador para PowerPoint (PPTX) - PENDIENTE DE IMPLEMENTACIÃ“N
"""
from pathlib import Path


# from pptx import Presentation  # pip install python-pptx

class PPTXTransformer:
    """
    Transforma presentaciones PowerPoint a texto plano

    TODO: Implementar cuando sea necesario
    - Extraer texto de slides
    - Extraer notas del presentador
    - Manejar tablas y grÃ¡ficos
    - Opcionalmente extraer imÃ¡genes con OCR
    """

    def __init__(self):
        self.supported_extensions = {'.pptx', '.ppt'}

    def transform(self, file_path: Path) -> str:
        """
        Transforma PPTX a texto plano

        Placeholder para implementaciÃ³n futura
        """
        raise NotImplementedError(
            "La transformaciÃ³n de PPTX aÃºn no estÃ¡ implementada. "
            "Se implementarÃ¡ cuando sea necesario."
        )

    # Estructura propuesta para cuando se implemente:
    # def _extract_slide_text(self, slide) -> str:
    #     pass
    #
    # def _extract_notes(self, slide) -> str:
    #     pass
    #
    # def _extract_tables(self, slide) -> List[str]:
    #     pass
