-- name: CellExists :one
SELECT EXISTS(
    SELECT 1
    FROM cells
    WHERE hash = $1
);

-- name: GetRandomCell :one
WITH mean_scores AS (
    SELECT cs.cell_id AS cell_id, AVG(score) AS mean_score
    FROM cell_scores AS cs
    GROUP BY cs.cell_id
    ORDER BY cs.cell_id DESC
    LIMIT 100
), weights AS (
    SELECT c.id AS id, (100 / SQRT(c.visits + 1)) + SUM(ms.mean_score) AS weight
    FROM cells AS c
    JOIN mean_scores AS ms
    ON ms.cell_id = c.id
    GROUP BY c.id
), rand_pick AS (
    SELECT random() * (SELECT SUM(weight) FROM weights) pick
), rand_id AS (
    SELECT id
    FROM (
        SELECT id, SUM(weight) OVER (ORDER BY id) scaled_weight, pick
        FROM weights CROSS JOIN rand_pick
    ) q
    WHERE scaled_weight >= pick
    ORDER BY id
    LIMIT 1
)
SELECT c.id, action, max_no_ops, initial, state
FROM cells AS c
JOIN rand_id AS ri ON ri.id = c.id WHERE c.id = ri.id;

-- name: InsertCell :exec
INSERT INTO cells (
    hash, action, max_no_ops, initial, state
) VALUES (
    $1, $2, $3, $4, $5
);

-- name: InsertCellScore :exec
INSERT INTO cell_scores (
    cell_id, score
) VALUES (
    $1, $2
);
