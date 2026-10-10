import re
import yaml
from pathlib import Path
from tqdm import tqdm
from langchain_core.documents import Document

class Parser:
    def __init__(self, dir_path: str | Path) -> None:
        self.dir_path = Path(dir_path).expanduser()
        self.documents = []

    def get_relative_path(self, note_path: Path, dir_path: str | Path = '') -> str:
        dir_path = Path(dir_path).expanduser() if dir_path else self.dir_path
        return note_path.relative_to(dir_path).as_posix()

    def parse_frontmatter(self, text: str) -> tuple[str, dict]:
        '''
        Returns body without frontmatter and dict of frontmatter metadata
        '''
        if not text.startswith('---'):
            return text, {}

        parts = text.split('---', 2)
        if len(parts) != 3:
            return text, {}

        frontmatter = yaml.safe_load(parts[1])
        body = parts[2]

        tags = frontmatter.get('tags', [])
        if isinstance(tags, str):
            tags = tags.split(', ')
        annotation_target = frontmatter.get('annotation-target', '')

        frontmatter_metadata = {
            'tags': tags,
            'annotation-target': annotation_target
        }

        return body, frontmatter_metadata

    def load_notes(self, dir_path: str | Path = '') -> list[Document]:
        dir_path = Path(dir_path).expanduser() if dir_path else self.dir_path

        md_files = (
            p for p in dir_path.rglob('*.md')
            if not any(part.startswith('.') for part in p.parts)
            and not p.name.endswith('.excalidraw.md')
        )

        for path in tqdm(md_files, desc='Parsing obsidian vault', disable=True):
            text = path.read_text(
                encoding='utf-8'
            )
            text, frontmatter_metadata = self.parse_frontmatter(text)

            self.documents.append(
                Document(
                    page_content=text,
                    metadata={
                        'source': self.get_relative_path(path),
                        'title': path.stem,
                        # 'links': links,
                        **frontmatter_metadata
                    },
                )
            )

        return self.documents

    def parse_note(self, path: Path) -> Document:
        text = path.read_text(encoding='utf-8')
        text, frontmatter_metadata = self.parse_frontmatter(text)

        return Document(
            page_content=text,
            metadata={
                # TODO
            }
        )

    def get_documents(self) -> list[Document]:
        return self.documents

    def __getitem__(self, idx):
        return self.documents[idx]

    # TESTING function. Do not use for retrieval.
    def get_note(self, source):
        for doc in self.documents:
            if doc.metadata.get('source') == source:
                return doc

if __name__ == '__main__':
    parser = Parser('.')
    parser.load_notes()
    print(parser.documents)
