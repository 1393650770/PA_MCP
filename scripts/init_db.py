# [AI:BEGIN]
# PA_MCP - Database Initialization Script
# Creates DuckDB database and initializes all tables.
# [AI:END]

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


async def main() -> None:
    from pa_mcp.data.store import DuckDBStore
    from pa_mcp.config import get_settings

    settings = get_settings()
    db_path = settings.database.path
    print(f"Initializing PA_MCP database at: {db_path}")

    store = DuckDBStore(db_path)
    store.connect()

    # List all tables
    tables = store.query_df(
        "SELECT table_name FROM information_schema.tables WHERE table_schema='main'",
    )
    print(f"\nCreated tables ({len(tables)}):")
    for _, row in tables.iterrows():
        print(f"  - {row['table_name']}")

    store.close()
    print("\nDatabase initialized. Run 'python -m pa_mcp.data.scheduler' for first data load.")


if __name__ == "__main__":
    asyncio.run(main())
