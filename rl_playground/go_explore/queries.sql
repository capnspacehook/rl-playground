-- name: CellExists :one
SELECT EXISTS(
    SELECT 1
    FROM cells
    WHERE hash = $1
);

-- name: GetRandomCell :one
WITH mean_scores AS (
    -- get mean of last 100 scores of all cells
    SELECT cell_id, AVG(score) AS mean_score
    FROM (
        SELECT cell_id, score, ROW_NUMBER() OVER (PARTITION BY cell_id ORDER BY id DESC) AS rn
        FROM cell_scores
    ) AS desc_scores
    WHERE rn <= 100
    GROUP BY cell_id
), weights AS (
    -- create weights for each cell based on number of visits and mean score
    -- less visits and a lower mean score results in a higher weight
    -- mean score is prioritized over number of visits
    SELECT 
        c.id AS id,
        (100 / SQRT(c.visits + 1)) + (
            (
                SELECT MAX(mean_score) AS max_score
                FROM mean_scores
            ) - SUM(ms.mean_score)
        ) AS weight
    FROM cells AS c
    JOIN mean_scores AS ms
    ON ms.cell_id = c.id
    GROUP BY c.id
), rand_pick AS (
    -- create value that will be used to pick a random cell
    -- multiply random number by sum of all weights so the weights don't have to add up to 100
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

-- name: GetFirstCell :one
SELECT id, action, max_no_ops, initial, state
FROM cells
ORDER BY id
LIMIT 1;

-- name: InsertCell :one
INSERT INTO cells (
    hash, action, max_no_ops, initial, state
) VALUES (
    $1, $2, $3, $4, $5
)
ON CONFLICT DO NOTHING
RETURNING id;

-- name: InsertCellScore :exec
INSERT INTO cell_scores (
    cell_id, score
) VALUES (
    $1, $2
);

-- name: IncrementCellVisit :exec
UPDATE cells
SET visits = visits + 1
WHERE id = $1;
