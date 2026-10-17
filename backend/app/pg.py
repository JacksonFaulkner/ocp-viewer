from contextlib import contextmanager

import psycopg

from app import config


@contextmanager
def connect():
    with psycopg.connect(config.DATABASE_URL) as c:
        yield c


def init_schema() -> None:
    sql = (config.ROOT / "sql" / "001_init.sql").read_text()
    with connect() as c:
        c.execute(sql)
