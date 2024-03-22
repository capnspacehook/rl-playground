from typing import Tuple

import sqlalchemy

from rl_playground.go_explore.queries import Querier


class StateManager(object):
    def __init__(self, engine: sqlalchemy.Engine) -> None:
        self.engine = engine

    def insert_initial_cell(self, hash: str, max_no_ops: int | None, state: memoryview):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.insert_cell(hash, None, max_no_ops, True, state)
            conn.commit()

    def is_cell_new(self, hash: str) -> bool:
        with self.engine.connect() as conn:
            q = Querier(conn)
            return q.cell_exists(hash)

    def insert_cell(self, hash: str, action: int, max_no_ops: int | None, state: memoryview):
        with self.engine.connect() as conn:
            q = Querier(conn)
            q.insert_cell(hash, action, max_no_ops, False, state)
            conn.commit()

    def get_random_cell(self) -> Tuple[int, int, int, bool, memoryview]:
        with self.engine.connect() as conn:
            q = Querier(conn)
            id, action, max_no_ops, initial, state = q.get_random_cell()
            return id, action, max_no_ops, initial, state
