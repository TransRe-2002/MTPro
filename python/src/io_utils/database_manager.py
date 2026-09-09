import sqlite3
from typing import Any

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path: str = db_path
        self.connection: sqlite3.Connection | None = None

    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)

    def disconnect(self):
        if self.connection is not None:
            self.connection.close()

    def execute_query(self, query: str, params=()) -> sqlite3.Cursor:
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor

    def execute_read_query(self, query: str, params=()) -> list[tuple]:
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
