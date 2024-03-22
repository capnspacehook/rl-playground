CREATE TABLE cells (
    id         SERIAL  PRIMARY KEY,
    hash       TEXT    UNIQUE NOT NULL,
    action     INTEGER,
    max_no_ops INTEGER,
    initial    BOOLEAN NOT NULL,
    visits     INTEGER NOT NULL DEFAULT 0,
    state      BYTEA   NOT NULL
);

CREATE TABLE cell_scores (
    id      SERIAL         PRIMARY KEY,
    cell_id INTEGER        NOT NULL REFERENCES cells(id),
    score   NUMERIC(10, 5) NOT NULL
);
