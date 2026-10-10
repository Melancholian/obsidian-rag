from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tomllib

@dataclass(frozen=True)
class Config:
    obsidian_path: Path
    lance_uri: Path
    lance_table: str
    graphml_path: Path
    hf_embeddings: str
    loader_batch_size: int # May be another batch_size settings

    @classmethod
    def load(cls, config_path: str | Path = 'config.toml') -> Config:
        with open(config_path, 'rb') as f:
            config = tomllib.load(f)

        raw_values = {
            'obsidian_path': config.get('obsidian', {}).get('obsidian_path'),
            'loader_batch_size': config.get('loader', {}).get('batch_size'),
            'hf_embeddings': config.get('lance', {}).get('hf_embeddings'),
            'lance_uri': config.get('lance', {}).get('lance_uri'),
            'lance_table': config.get('lance', {}).get('lance_table'),
            'graphml_path': config.get('graph', {}).get('graphml_path')
        }
        missing_fields = [
            field for field, value in raw_values.items() if value is None
        ]
        if missing_fields:
            raise ValueError(
                f'Missing fields in configuration file: {', '.join(missing_fields)}'
            )

        return cls(
            obsidian_path=Path(raw_values['obsidian_path']).expanduser(),
            lance_uri=Path(raw_values['lance_uri']).expanduser(),
            lance_table=raw_values['lance_table'],
            graphml_path=Path(raw_values['graphml_path']).expanduser(),
            hf_embeddings=raw_values['hf_embeddings'],
            loader_batch_size=int(raw_values['loader_batch_size']) # error catch?
        )
