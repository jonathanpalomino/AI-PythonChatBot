import unittest
from unittest.mock import MagicMock
from io import BytesIO
from src.services.pdf_service import PDFService
from src.models.models import MessageRole

class TestPDFService(unittest.TestCase):
    def setUp(self):
        self.pdf_service = PDFService()

    def test_clean_markdown(self):
        text = "**bold** *italic* `code` [link](url) ```block```"
        cleaned = self.pdf_service._clean_markdown(text)
        self.assertIn("bold", cleaned)
        self.assertIn("italic", cleaned)
        self.assertIn("code", cleaned)
        self.assertIn("link", cleaned)
        self.assertIn("[Code Block]", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("```", cleaned)

    def test_generate_pdf(self):
        messages = [
            MagicMock(role=MessageRole.USER, content="Hello"),
            MagicMock(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
        
        pdf_buffer = self.pdf_service.generate_conversation_pdf("Test Title", messages)
        self.assertIsInstance(pdf_buffer, BytesIO)
        self.assertGreater(len(pdf_buffer.getvalue()), 0)
        # Check PDF header bytes
        self.assertTrue(pdf_buffer.getvalue().startswith(b"%PDF"))

if __name__ == "__main__":
    unittest.main()
