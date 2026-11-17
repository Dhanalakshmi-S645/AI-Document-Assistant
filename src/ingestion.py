
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_HEADINGS = {
    "SUMMARY",
    "PROJECT",
    "EDUCATION",
    "COURSEWORK",
    "SKILLS",
}


def extract_text(pdf_path):
    """Extract text from every page of a PDF."""

    reader = PdfReader(pdf_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def split_into_blocks(text):
    """
    Divide document text into logical sections.
    Preserve section names and individual project names.
    """

    lines = text.splitlines()

    sections = []
    current_heading = "DOCUMENT"
    current_lines = []

    for line in lines:

        cleaned = line.strip()

        if cleaned.upper() in SECTION_HEADINGS:

            if current_lines:
                sections.append((
                    current_heading,
                    current_lines
                ))

            current_heading = cleaned.upper()
            current_lines = []

        else:
            current_lines.append(line)

    if current_lines:
        sections.append((
            current_heading,
            current_lines
        ))

    blocks = []

    for heading, section_lines in sections:

        if heading == "PROJECT":

            project_lines = []
            project_name = "PROJECT"

            for index, line in enumerate(section_lines):

                cleaned = line.strip()

                # A project title is followed by its GitHub link.
                is_project_title = (
                    cleaned
                    and index + 1 < len(section_lines)
                    and "github.com/" in section_lines[index + 1].lower()
                    and not cleaned.startswith(("•", "-", "*"))
                )

                if is_project_title and project_lines:

                    content = "\n".join(project_lines).strip()

                    if content:
                        blocks.append({
                            "section": project_name,
                            "text": content
                        })

                    project_lines = []

                if is_project_title:
                    project_name = cleaned

                project_lines.append(line)

            if project_lines:

                content = "\n".join(project_lines).strip()

                if content:
                    blocks.append({
                        "section": project_name,
                        "text": content
                    })

        else:

            content = "\n".join(section_lines).strip()

            if content:
                blocks.append({
                    "section": heading,
                    "text": f"{heading}\n{content}"
                })

    return blocks


def chunk_text(text, chunk_size=700, overlap=100):
    """Split text while preferring paragraph and line boundaries."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    return splitter.split_text(text)


def process_pdf(pdf_path):
    """Extract PDF text and create section-aware chunks."""

    pages = extract_text(pdf_path)

    all_chunks = []

    for page in pages:

        blocks = split_into_blocks(page["text"])

        for block in blocks:

            chunks = chunk_text(block["text"])

            for chunk in chunks:

                all_chunks.append({
                    "text": chunk,
                    "page": page["page"],
                    "chunk": len(all_chunks),
                    "section": block["section"]
                })

    return all_chunks
