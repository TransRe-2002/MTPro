import sqlite3
from typing import Any, Optional

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path: str = db_path
        self.connection: Optional[sqlite3.Connection] = None

    def connect(self):
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)

    def disconnect(self):
        if self.connection is not None:
            self.connection.close()

    def execute_query(self, query: str, params=()) -> Optional[Any]:
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor

    def execute_read_query(self, query: str, params=()) -> Optional[Any]:
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        return cursor.fetchall()
