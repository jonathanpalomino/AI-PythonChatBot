# =============================================================================
# src/utils/text_utils.py
# Text Processing Utilities
# =============================================================================
import re

def clean_markdown_text(text: str) -> str:
    """
    Basic cleaning of markdown for PDF output and Unicode safety.
    
    Args:
        text (str): The input markdown text.
        
    Returns:
        str: The cleaned text safe for PDF generation.
    """
    if not text:
        return ""
    
    # 1. Standard Markdown Cleaning
    # Remove bold markers (**, __)
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    
    # Remove italic markers (*)
    # We avoid stripping single underscores (_) because they are common in technical names
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    
    # Optional: only strip underscores if they are likely intended as italics (surrounded by spaces)
    text = re.sub(r'(^|\s)_([^_]+)_(\s|$)', r'\1\2\3', text)
    
    # Remove code blocks and inline code
    text = re.sub(r'```.*?```', '[Code Block]', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    
    # Remove markdown links
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    
    # 2. Unicode Normalization (Replace smart characters with ASCII equivalents)
    replacements = {
        '\u2018': "'", '\u2019': "'",  # Smart quotes
        '\u201c': '"', '\u201d': '"',  # Smart double quotes
        '\u2013': '-', '\u2014': '--', # Dashes
        '\u2026': '...',               # Ellipsis
        '\u00a0': ' ',                 # Non-breaking space
        '\u2022': '*',                 # Bullet point
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
        
    # 3. Final safety pass: replace any remaining non-latin-1 characters 
    # that Helvetica can't handle to prevent FPDFUnicodeEncodingException
    text = text.encode('latin-1', 'replace').decode('latin-1')
    
    # Replace newlines with consistent format
    text = text.replace('\r\n', '\n')
    
    return text
