"""Verify the packaged inference artifacts or recover them explicitly in development."""

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

from config import ARTIFACT_DIR

BASE = "https://huggingface.co/spaces/jesus1558/Nubira/resolve/main/"
DEST = ARTIFACT_DIR
ARTIFACTS = {
    "best_weather_risk_model.pt": (433623, "234e65463355ef7934fff6d20d60387c1fd173c855153e3ed54172e79544e42b"),
    "climate_context.npz": (19364134, "6e7ede881b621805537cd68feddc884534546991b10b33a1741864d54e49b5b3"),
}


def valid_artifact(path: Path, size: int, expected: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == expected


def verify():
    for name, (size, expected) in ARTIFACTS.items():
        if not valid_artifact(DEST / name, size, expected):
            raise RuntimeError(f"Artefacto local ausente o inválido: {name}.")
        print(f"{name}: verificado.")


def fetch():
    DEST.mkdir(exist_ok=True)
    for name, (size, expected) in ARTIFACTS.items():
        destination = DEST / name
        if valid_artifact(destination, size, expected):
            print(f"{name}: verificado.")
            continue
        digest = hashlib.sha256()
        downloaded = 0
        with tempfile.NamedTemporaryFile(dir=DEST, delete=False) as output:
            temporary = Path(output.name)
            try:
                with urlopen(BASE + name, timeout=30) as response:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        downloaded += len(chunk)
                        if downloaded > size:
                            raise RuntimeError(f"Tamaño inesperado de {name}.")
                        digest.update(chunk)
                        output.write(chunk)
                if downloaded != size or digest.hexdigest() != expected:
                    raise RuntimeError(f"Checksum inválido de {name}; se conserva la copia anterior.")
                output.close()
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"{name}: descargado y verificado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verifica los archivos locales sin usar la red.")
    verify() if parser.parse_args().check else fetch()
