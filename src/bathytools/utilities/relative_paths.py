from pathlib import Path

MAIN_DIR = Path(__file__).parent.parent.parent.resolve().parent.resolve()


def read_path(raw_path: str) -> Path:
    if "${MAIN_DIR}" in raw_path:
        raw_path = raw_path.replace("${MAIN_DIR}", str(MAIN_DIR))
    return Path(raw_path).resolve()
