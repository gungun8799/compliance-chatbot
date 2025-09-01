import os
import glob
import re
from llama_index.readers.file import DocxReader

def section_aware_split(text: str, max_chunk_len: int = 1500) -> list:
    """
    Chunk a Markdown-style document into hierarchical sections (using #, ##, ###)
    and return structured chunks with section path and level.
    """
    lines = text.splitlines()
    chunks = []
    current_chunk_lines = []
    current_path = []

    def flush_chunk():
        if not current_chunk_lines:
            return
        content = "\n".join(current_chunk_lines).strip()
        if content:
            chunks.append({
                "section_path": current_path.copy(),
                "level": len(current_path),
                "content": content
            })

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if header_match:
            flush_chunk()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_path = current_path[: level - 1] + [title]
            current_chunk_lines = [line]
        else:
            current_chunk_lines.append(line)

    flush_chunk()
    return chunks

def get_all_docx_text(docx_folder: str) -> str:
    """
    Load all .docx files under `docx_folder`, extract their text via DocxReader,
    and concatenate into one large string.
    """
    reader = DocxReader()
    texts = []
    pattern = os.path.join(docx_folder, "*.docx")
    for file_path in glob.glob(pattern):
        pages = reader.load_data(file_path)
        for page in pages:
            texts.append(page.text)
    return "\n\n".join(texts)