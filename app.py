"""AutoDetect2 — entry point.

    python app.py                 # http://127.0.0.1:8000
    python app.py --port 9000 --reload
"""

from __future__ import annotations

import argparse

from autodetect2.backend.main import app  # noqa: F401 - re-exported for `uvicorn app:app`


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="AutoDetect2 web workbench")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "autodetect2.backend.main:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
