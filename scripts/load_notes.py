from ingestion import Loader
from core import Config

LANCEDB_URI = 'data/lancedb'

if __name__ == '__main__':
    cfg = Config.load()
    loader = Loader.from_cfg(cfg)
    # loader.load()

