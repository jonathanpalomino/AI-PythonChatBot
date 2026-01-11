# =============================================================================
# src/document_loaders/html_loader.py
# =============================================================================
"""
Loader para archivos HTML
"""
import re
from pathlib import Path
from typing import List, Dict, Any

from bs4 import BeautifulSoup, Comment

from .base_loader import BaseDocumentLoader, DocumentSection, ProcessedDocument


class HTMLLoader(BaseDocumentLoader):
    """Carga y procesa archivos HTML"""

    def __init__(self, remove_scripts: bool = True,
                 remove_styles: bool = True,
                 remove_comments: bool = True):
        super().__init__()
        self.supported_extensions = {'.html', '.htm'}
        self.remove_scripts = remove_scripts
        self.remove_styles = remove_styles
        self.remove_comments = remove_comments

    def load(self, file_path: Path, original_filename: str = None) -> ProcessedDocument:
        """Carga un archivo HTML"""
        # Try different encodings
        content = None
        encoding_used = 'utf-8'

        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            raise ValueError(f"No se pudo decodificar el archivo {file_path}")

        # Parse HTML
        soup = BeautifulSoup(content, 'html.parser')

        # Clean HTML
        self._clean_html(soup)

        # Extract metadata
        metadata = self._extract_metadata(soup, encoding_used)

        # Extract sections
        sections = self.extract_sections(soup)

        # Generate full content
        full_content = self._generate_full_content(sections)

        # Convertir a ruta relativa
        abs_path = file_path if file_path.is_absolute() else file_path.resolve()
        try:
            relative_path = abs_path.relative_to(Path.cwd())
        except ValueError:
            relative_path = abs_path

        return ProcessedDocument(
            file_path=str(relative_path),
            file_name=file_path.name,
                original_filename=original_filename or file_path.name,
            content=full_content,
            sections=sections,
            metadata=metadata
        )

    def _clean_html(self, soup: BeautifulSoup):
        """Remove unwanted elements from HTML"""
        # Remove scripts
        if self.remove_scripts:
            for script in soup.find_all('script'):
                script.decompose()

        # Remove styles
        if self.remove_styles:
            for style in soup.find_all('style'):
                style.decompose()

        # Remove comments
        if self.remove_comments:
            for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
                comment.extract()

        # Remove common navigation/footer elements
        for tag in soup.find_all(['nav', 'footer', 'aside']):
            tag.decompose()

    def _extract_metadata(self, soup: BeautifulSoup, encoding: str) -> Dict[str, Any]:
        """Extract metadata from HTML"""
        metadata = {
            'encoding': encoding
        }

        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()

        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name') or meta.get('property', '')
            content = meta.get('content', '')

            if name and content:
                # Common meta tags
                if name.lower() in ['description', 'keywords', 'author',
                                    'og:title', 'og:description']:
                    metadata[name.lower().replace(':', '_')] = content

        # Language
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')

        return metadata

    def extract_sections(self, soup: BeautifulSoup) -> List[DocumentSection]:
        """
        Extract sections from HTML based on semantic structure
        (h1, h2, h3, article, section tags)
        """
        sections = []

        # Try to find main content area
        main_content = soup.find('main') or soup.find('article') or soup.find('body')

        if not main_content:
            # Fallback: entire document
            sections.append(DocumentSection(
                title="HTML Content",
                content=soup.get_text().strip(),
                level=1,
                metadata={}
            ))
            return sections

        # Extract by headings
        current_section = None
        current_content = []
        current_level = 0
        pre_heading_content = []

        for element in main_content.descendants:
            # Skip non-tag elements that are just whitespace
            if isinstance(element, str) and not element.strip():
                continue

            # Check if it's a heading
            if hasattr(element, 'name') and element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                heading_text = element.get_text().strip()

                if not heading_text:
                    continue

                # Save pre-heading content
                if current_section is None and pre_heading_content:
                    intro_content = self._clean_text('\n'.join(pre_heading_content))
                    if intro_content:
                        sections.append(DocumentSection(
                            title="Introduction",
                            content=intro_content,
                            level=0,
                            metadata={}
                        ))
                    pre_heading_content = []

                # Save previous section
                if current_section:
                    section_content = self._clean_text('\n'.join(current_content))
                    if section_content:
                        sections.append(DocumentSection(
                            title=current_section,
                            content=section_content,
                            level=current_level,
                            metadata={}
                        ))

                # New section
                current_level = int(element.name[1])  # h1 -> 1, h2 -> 2, etc.
                current_section = heading_text
                current_content = []

            # Extract text from paragraph-like elements
            elif hasattr(element, 'name') and element.name in ['p', 'div', 'li', 'td', 'th']:
                text = element.get_text().strip()
                if text:
                    if current_section:
                        current_content.append(text)
                    else:
                        pre_heading_content.append(text)

        # Save pre-heading content if no sections found
        if current_section is None and pre_heading_content:
            intro_content = self._clean_text('\n'.join(pre_heading_content))
            if intro_content:
                sections.append(DocumentSection(
                    title="HTML Content",
                    content=intro_content,
                    level=0,
                    metadata={}
                ))

        # Save last section
        if current_section:
            section_content = self._clean_text('\n'.join(current_content))
            if section_content:
                sections.append(DocumentSection(
                    title=current_section,
                    content=section_content,
                    level=current_level,
                    metadata={}
                ))

        # Handle tables separately if found
        tables = main_content.find_all('table')
        for i, table in enumerate(tables, 1):
            table_content = self._extract_table_content(table)
            if table_content:
                sections.append(DocumentSection(
                    title=f"Table {i}",
                    content=table_content,
                    level=2,
                    metadata={'type': 'table'}
                ))

        return sections

    def _extract_table_content(self, table) -> str:
        """Extract content from HTML table"""
        rows = []

        for row in table.find_all('tr'):
            cells = row.find_all(['td', 'th'])
            row_text = ' | '.join(cell.get_text().strip() for cell in cells)
            if row_text.strip():
                rows.append(row_text)

        return '\n'.join(rows) if rows else ''

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _generate_full_content(self, sections: List[DocumentSection]) -> str:
        """Generate full content from sections"""
        full_content = []

        full_content.append("# HTML Document")
        full_content.append("")

        for section in sections:
            # Add section header
            header_prefix = '#' * (section.level + 1 if section.level > 0 else 2)
            full_content.append(f"{header_prefix} {section.title}")
            full_content.append(section.content)
            full_content.append("")

        return '\n'.join(full_content).strip()
