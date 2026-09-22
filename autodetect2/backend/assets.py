from __future__ import annotations

import re
from pathlib import Path

from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles

from .config import STATIC_DIR

# `from "./x.js"`, `from "../y/z.js"`, `import("./w.js")`
SPECIFIER = re.compile(r"""((?:from|import)\s*\(?\s*["'])(\.{1,2}/[^"']+?\.js)(["'])""")


def _stamp(path: Path) -> int:
    try:
        return int(path.stat().st_mtime)
    except OSError:
        return 0


def version_imports(source: str, module: Path) -> str:
    """Rewrites relative imports so every module in the graph carries its mtime.

    Stamping only the entry point is not enough: a page can load `page.js?v=2`
    whose `import "./widget.js"` still resolves to the cached copy from an hour
    ago, and the mismatch surfaces as a missing method on a fresh object. Every
    edge of the import graph has to be versioned, so an edited file invalidates
    itself and nothing else.
    """

    def replace(match: re.Match) -> str:
        head, specifier, tail = match.groups()
        target = (module.parent / specifier).resolve()
        return f"{head}{specifier}?v={_stamp(target)}{tail}"

    return SPECIFIER.sub(replace, source)


class VersionedStatic(StaticFiles):
    """Static files that never go stale.

    JavaScript is rewritten on the way out and served with `no-cache`; everything
    else is passed through untouched.
    """

    def file_response(self, full_path, stat_result, scope, status_code=200):
        path = Path(full_path)
        if path.suffix != ".js":
            response = super().file_response(full_path, stat_result, scope, status_code)
            response.headers["Cache-Control"] = "no-cache"
            return response

        # Deliberately not memoised: the output depends on the mtimes of the
        # imports, not only of this file, so a cache keyed by this file alone
        # would hand out stale links exactly when a dependency was edited.
        body = version_imports(path.read_text(encoding="utf-8"), path.resolve()).encode("utf-8")
        stamp = int(stat_result.st_mtime)

        return Response(
            content=body,
            media_type="text/javascript; charset=utf-8",
            headers={"Cache-Control": "no-cache", "ETag": f'W/"{stamp}-{len(body)}"'},
        )


def asset(path: str) -> str:
    """Static URL stamped with the file's mtime, for use from templates."""
    return f"/static/{path}?v={_stamp(STATIC_DIR / path)}"


__all__ = ["VersionedStatic", "asset", "version_imports"]
