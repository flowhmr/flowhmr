"""Fetch the official, pure-PyTorch subset needed for Pi4 camera inference."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import tempfile
from urllib.request import Request, urlopen

REVISION = "32dfbea9ce6268ea17446208a804710df9475c32"
DESTINATION = Path(__file__).resolve().parents[1] / "third_party" / "pi4_camera"
FILES = ("LICENSE", "LICENSE_MODEL", "NOTICE", "licenses/DINOV3.txt", "licenses/PI3.txt", "licenses/MOGE.txt",
         "licenses/DEPTH_ANYTHING_V2.txt", "licenses/APACHE-2.0.txt", "pi4/models/encoder.py", "pi4/models/decoder.py",
         "pi4/models/layers.py", "pi4/models/heads/camera.py", "pi4/models/heads/metric.py")


def main():
    missing = FILES
    installed = DESTINATION.exists()
    if installed:
        revision = DESTINATION / "REVISION"
        if not revision.is_file() or revision.read_text().strip() != REVISION:
            raise RuntimeError(f"Different Pi4 sources already exist in {DESTINATION}")
        missing = [name for name in FILES if not (DESTINATION / name.removeprefix("pi4/models/")).is_file()]
        if not missing:
            return
    with tempfile.TemporaryDirectory(dir=DESTINATION.parent) as tmp:
        package = Path(tmp) / "pi4_camera"

        def fetch(name):
            path = package / name.removeprefix("pi4/models/")
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://api.github.com/repos/yyfz/Pi4/contents/{name}?ref={REVISION}"
            req = Request(url, headers={"Accept": "application/vnd.github.raw+json"})
            with urlopen(req, timeout=60) as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(fetch, missing))
        if installed:
            for name in missing:
                path = Path(name.removeprefix("pi4/models/"))
                (DESTINATION / path).parent.mkdir(parents=True, exist_ok=True)
                (package / path).replace(DESTINATION / path)
        else:
            (package / "REVISION").write_text(REVISION + "\n")
            package.rename(DESTINATION)


if __name__ == "__main__":
    main()
