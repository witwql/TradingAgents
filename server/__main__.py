"""Launch the dashboard: python -m server [--host 127.0.0.1] [--port 8000]"""

import argparse

import uvicorn

from .app import create_app
from .db import Database


def main():
    parser = argparse.ArgumentParser(description="TradingAgents Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--db", default=None, help="SQLite file path override")
    args = parser.parse_args()

    import os as _os

    db = Database(args.db) if args.db else Database()
    override = _os.environ.get("TRADINGAGENTS_DASHBOARD_HOME")
    if override:
        print(f"\033[33m⚠ TRADINGAGENTS_DASHBOARD_HOME={override} —— 本次数据读写此目录，"
              f"与默认库(~/.tradingagents/dashboard)相互独立！\033[0m")
    try:
        favs = db.fetchone("SELECT COUNT(*) AS n FROM favorites")["n"]
        done = db.fetchone("SELECT COUNT(*) AS n FROM tasks WHERE status='completed'")["n"]
        print(f"数据库: {db._conn.execute('PRAGMA database_list').fetchone()[2]}")
        print(f"现有数据: 自选股 {favs} 条 · 已完成任务 {done} 条")
    except Exception:
        pass
    app = create_app(db=db, args_db_path=args.db)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
