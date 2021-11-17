"""Script to build the documentation."""

import os
import shutil
from pathlib import Path
from subprocess import run

ROOT_DIR = Path(__file__).parent.parent
EXCLUDE_PRIVATE = "_source,_asdict,_fields,_field_defaults,_field_types,_replace,_make"


def _main():
    api_dir = ROOT_DIR / "doc/api"
    shutil.rmtree(api_dir, ignore_errors=True)

    print("Generating documentation from docstrings")
    run(
        ["sphinx-apidoc", "-M", "-o", str(api_dir), "slurmer"],
        env={
            "SPHINX_APIDOC_OPTIONS": "members,undoc-members,show-inheritance,"
            "inherited-members,change-1",
            **os.environ.copy(),
        },
    )

    for file in api_dir.glob("*.rst"):
        with file.open("r+t") as f:
            data = f.read()
            data = data.replace("change-1", f"exclude-members: {EXCLUDE_PRIVATE}")
            f.seek(0)
            f.write(data)

    print("Building documentation")
    run(
        [
            "sphinx-build",
            "-M",
            "html",
            str(ROOT_DIR / "doc"),
            str(ROOT_DIR / "doc/build"),
        ]
    )


if __name__ == "__main__":
    _main()
