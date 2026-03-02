# =============================================================================
# src/services/pdf_service.py
# PDF Export Service for Conversations
# =============================================================================
from io import BytesIO
from typing import List

from fpdf import FPDF

from src.utils.logger import get_logger

logger = get_logger(__name__)

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "Conversation Export", border=False, align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

from src.utils.text_utils import clean_markdown_text
from src.utils.date_utils import get_current_utc


class PDFService:
    """Service to handle PDF generation for conversations"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def generate_conversation_pdf(self, title: str, messages: List[any]) -> BytesIO:
        """ Generates a PDF from a list of messages """
        self.logger.info(f"Generating PDF for conversation: {title}")

        pdf = PDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Title section
        pdf.set_font("helvetica", "B", 16)
        pdf.multi_cell(0, 10, title)
        pdf.set_font("helvetica", "", 10)
        pdf.cell(0, 10, f"Exported on: {get_current_utc().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = clean_markdown_text(msg.content)

            # Role header
            pdf.set_font("helvetica", "B", 12)
            if role.lower() == "user":
                pdf.set_text_color(0, 51, 102) # Dark blue
                pdf.cell(0, 10, "User:", ln=True)
            else:
                pdf.set_text_color(0, 102, 51) # Dark green
                pdf.cell(0, 10, "AI Assistant:", ln=True)

            # Message content
            pdf.set_text_color(0, 0, 0) # black
            pdf.set_font("helvetica", "", 11)

            # fpdf2 handles UTF-8 by default. We don't need the latin-1 hacks anymore.
            pdf.multi_cell(0, 7, content)

            pdf.ln(5)
            # Subtle separator
            pdf.set_draw_color(230, 230, 230)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(5)

        # Modern fpdf2 returns bytes. If it returns str, it means the legacy fpdf is still installed.
        pdf_bytes = pdf.output()

        if isinstance(pdf_bytes, str):
            raise RuntimeError(
                "ERROR: Se detectó la versión antigua de 'fpdf'. "
                "Por favor, desinstálala ejecutando: pip uninstall fpdf"
            )

        pdf_output = BytesIO()
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)

        return pdf_output

pdf_service = PDFService()
