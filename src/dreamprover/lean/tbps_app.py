"""Configure the upstream TBPS API without editing its checkout."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


backend = Path(os.environ["TBPS_BACKEND_DIR"])
sys.path.insert(0, str(backend))

from base_server import TheoremResult, create_app  # noqa: E402
from search_app.WL_embedding.db_utils import DB_CONFIG, connect_to_db  # noqa: E402
from search_app.cse import cse  # noqa: E402
from search_app.myexpr import deserialize_expr  # noqa: E402
from search_app.process_single import process_single_prop_new  # noqa: E402

# Upstream modules share this mapping, including connect_to_db's default arg.
DB_CONFIG.update(
    host=os.environ.get("PGHOST", "127.0.0.1"),
    port=int(os.environ.get("PGPORT", "5432")),
    database=os.environ.get("PGDATABASE", "mathlib_db"),
    user=os.environ.get("PGUSER", "postgres"),
    password=os.environ.get("PGPASSWORD", ""),
    connect_timeout=5,
)


class TBPSHandler:
    def __init__(self) -> None:
        self.lean = os.environ.get("TBPS_LEAN", "lean")
        self.parser_dir = Path(os.environ["TBPS_PARSER_DIR"])
        self.imports = [
            name for name in os.environ.get("TBPS_LEAN_IMPORTS", "").split(",") if name
        ]
        for name in self.imports:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*", name):
                raise ValueError(f"Invalid Lean module name: {name!r}")
        self.env = os.environ.copy()
        self.env["LEAN_PATH"] = os.pathsep.join(
            filter(
                None,
                [
                    str(self.parser_dir),
                    os.environ.get("TBPS_LEAN_PATH"),
                    self.env.get("LEAN_PATH"),
                ],
            )
        )
        self.lock = asyncio.Lock()

    def parse(self, expression: str) -> dict:
        with tempfile.TemporaryDirectory(prefix="dreamprover-tbps-") as directory:
            path = Path(directory)
            (path / "input.txt").write_text(expression, encoding="utf-8")
            imports = "".join(f"import {name}\n" for name in self.imports)
            (path / "Parse.lean").write_text(
                "import Mathlib_Construction\n"
                + imports
                + 'set_option maxRecDepth 100000\nparse_and_write "input.txt" "output.json"\n',
                encoding="utf-8",
            )
            result = subprocess.run(
                [self.lean, "Parse.lean"],
                cwd=path,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode:
                raise ValueError(
                    f"Lean expression could not be parsed: {result.stdout or result.stderr}"
                )
            return json.loads((path / "output.json").read_text(encoding="utf-8"))

    def search(self, expression: str, k: int) -> tuple[list[TheoremResult], str]:
        parsed = self.parse(expression)
        results = process_single_prop_new(cse(deserialize_expr(parsed["your_expr"])), k)
        return [
            TheoremResult(
                name=name,
                similarity_score=round(score, 4),
                statement=statement,
                node_count=nodes,
            )
            for name, score, statement, nodes in results
        ], parsed["expr_dbg"]

    async def find_similar_theorems(
        self, expression: str, k: int, node_ratio: float | None = None
    ):
        if not expression.strip():
            raise ValueError("Expression cannot be empty")
        if node_ratio is not None:
            raise ValueError("This upstream search selects node_ratio automatically")
        async with self.lock:
            return await asyncio.to_thread(self.search, expression, k)

    def health(self) -> tuple[bool, bool, str]:
        database_ok = False
        try:
            conn = connect_to_db()
            if conn is not None:
                with conn, conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT to_regclass('mathlib_filtered'), to_regclass('wl_encodings_new')"
                    )
                    database_ok = all(cursor.fetchone())
                conn.close()
        except Exception:
            database_ok = False
        try:
            self.parse("True")
            lean_ok = True
        except Exception:
            lean_ok = False
        return database_ok, lean_ok, "dreamprover-tbps-adapter-1"

    async def check_health(self):
        return await asyncio.to_thread(self.health)


app = create_app(
    TBPSHandler(),
    "TBPS for DreamProver",
    "Upstream tree-based retrieval with configurable Lean and PostgreSQL paths",
)
