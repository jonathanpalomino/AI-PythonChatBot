# =============================================================================
# src/services/pdf_service.py
# PDF Export Service for Conversations
# =============================================================================
from datetime import datetime
from io import BytesIO
from typing import List
import re

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

class PDFService:
    """Service to handle PDF generation for conversations"""

    def __init__(self):
        self.logger = get_logger(__name__)

    def _clean_markdown(self, text: str) -> str:
        """Basic cleaning of markdown for PDF output"""
        if not text:
            return ""
        
        # Remove bold/italic markers
        text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text)
        text = re.sub(r'(\*|_)(.*?)\1', r'\2', text)
        
        # Remove code blocks and inline code
        text = re.sub(r'```.*?```', '[Code Block]', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        # Remove markdown links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        
        # Replace newlines with consistent format
        text = text.replace('\r\n', '\n')
        
        return text

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
        pdf.cell(0, 10, f"Exported on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        for msg in messages:
            role = msg.role.value if hasattr(msg.role, 'value') else str(msg.role)
            content = self._clean_markdown(msg.content)
            
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
            
            # Use multi_cell for wrapping text
            # fpdf2 handles UTF-8 by default if using core fonts with latin-1 or by adding TTF fonts
            # For simplicity and to avoid missing font errors, we'll try to use standard fonts 
            # and replace common problematic chars if needed, though fpdf2 is generally better at this.
            try:
                # Replace common non-latin-1 chars for safety if not using unicode fonts
                # fpdf2 usually handles this if we tell it to.
                content_safe = content.encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(0, 7, content_safe)
            except Exception as e:
                self.logger.warning(f"Error encoding PDF content: {e}")
                pdf.multi_cell(0, 7, content)
                
            pdf.ln(5)
            # Subtle separator
            pdf.set_draw_color(230, 230, 230)
            pdf.line(15, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(5)

        # Output to buffer
        pdf_output = BytesIO()
        # In fpdf2, output can return bytes or be written to a file/buffer
        pdf_bytes = pdf.output()
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        
        return pdf_output

pdf_service = PDFService()
