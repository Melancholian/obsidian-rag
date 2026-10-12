import re
from tqdm import tqdm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class Chunker:
    def __init__(self, chunk_size=1200, chunk_overlap=150) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.excluded_filetypes = ('.png', '.jpg', '.jpeg', '.gif', '.pdf')
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ('#', 'h1'),
                ('##', 'h2'),
                ('###', 'h3'),
                ('####', 'h4'),
                ('#####', 'h5'),
                ('######', 'h6'),
            ],
            strip_headers=False
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                '\n\n',
                '\n',
                '. ',
                ' ',
                '',
            ]
        )

    def extract_links(self, text: str) -> tuple[str, list[str]]:
        '''
        Returns body and extracted links list
        '''
        wikilink_rx = re.compile(
            r"\[\[(?P<target>[^|\]#]+)"
            r"(?:#[^|\]]+)?"
            r"(?:\|(?P<alias>[^\]]+))?"
            r"\]\]"
        )

        extracted_links = set()

        def replace_match(match: re.Match) -> str:
            target = match.group("target").strip()
            alias = match.group("alias")

            # Убираем вложения
            if not target.endswith(self.excluded_filetypes):
                extracted_links.add(target + '.md')

            return alias.strip() if alias else target

        clean_text = wikilink_rx.sub(replace_match, text)

        return clean_text, list(extracted_links)

    def split_document(self, document):
        total_chunks = []

        sections = self.markdown_splitter.split_text(
            document.page_content
        )

        for section in sections:
            section.metadata.update(
                document.metadata
            )

            chunks = self.text_splitter.split_documents([section])

            for chunk in chunks:
                clean_text, links = self.extract_links(chunk.page_content) # After chunking?
                chunk.page_content = clean_text
                chunk.metadata['links'] = links

                total_chunks.append(chunk)

        return total_chunks

    def split_documents(self, documents):
        total_chunks = []

        for document in tqdm(documents, desc='Chunking documents', disable=True):
            chunks = self.split_document(document)
            total_chunks.extend(chunks)

        return total_chunks
