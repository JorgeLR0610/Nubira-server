"""Recover the checked-in Git LFS artifacts from the project's public mirror."""

import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

BASE = "https://huggingface.co/spaces/jesus1558/Nubira/resolve/main/"
DEST = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS = {
    "best_weather_risk_model.pt": (433623, "234e65463355ef7934fff6d20d60387c1fd173c855153e3ed54172e79544e42b"),
    "climate_context.npz": (19364134, "6e7ede881b621805537cd68feddc884534546991b10b33a1741864d54e49b5b3"),
}


def fetch():
    DEST.mkdir(exist_ok=True)
    for name, (size, expected) in ARTIFACTS.items():
        destination = DEST / name
        checked_in = DEST.parent / name
        active = destination if destination.exists() else checked_in
        if (active.is_file() and active.stat().st_size == size and
                hashlib.sha256(active.read_bytes()).hexdigest() == expected):
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
    fetch()
