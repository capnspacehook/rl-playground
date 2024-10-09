-- name: CellExists :one
SELECT EXISTS(
    SELECT 1
    FROM cells
    WHERE hash = $1
);

-- name: GetRandomCell :one
WITH mean_scores AS (
    -- get mean of last 10 scores of all cells in certain sections
    SELECT cell_id, AVG(score) AS mean_score
    FROM (
        SELECT 
            cs.cell_id,
            cs.score,
            ROW_NUMBER() OVER (PARTITION BY cs.cell_id ORDER BY cs.id DESC) AS rn
        FROM cell_scores AS cs
        JOIN cells AS c
        ON c.id = cs.cell_id
        WHERE c.section <= $1 and c.invalid = FALSE
    ) AS q
    WHERE rn <= 10
    GROUP BY cell_id
), norm_scores AS (
    -- normalize mean scores to be between 0 and 100
    SELECT 
        cell_id,
        -- ensure we aren't dividing by 0
        ((mean_score - min_score) / COALESCE(NULLIF(max_score - min_score, 0), 1)) * 100 AS norm_score
    FROM (
        SELECT 
            cell_id,
            mean_score,
            MIN(mean_score) OVER () AS min_score,
            MAX(mean_score) OVER () AS max_score
        FROM mean_scores
    ) AS q
), weights AS (
    -- create weights for each cell based on number of visits and normalized score
    -- less visits and a lower normalized score results in a higher weight
    -- normalized score is prioritized over number of visits
    SELECT 
        c.id AS id,
        (100 / SQRT(c.visits + 1)) + (
            -- subtract normalized scores by max normalized score so
            -- cells with a greater normalized score have less weight
            (
                SELECT MAX(norm_score) AS max_score
                FROM norm_scores
            ) - SUM(ns.norm_score)
        ) +
        CASE
            -- add 5% of the max possible weight to cells in the current section
            WHEN c.section = $1 THEN 10
            ELSE 0 
        END AS weight
    FROM cells AS c
    JOIN norm_scores AS ns
    ON ns.cell_id = c.id
    GROUP BY c.id
), rand_pick AS (
    -- create value that will be used to pick a random cell
    -- multiply random number by sum of all weights so the weights don't have to add up to 100
    SELECT random() * (SELECT SUM(weight) FROM weights) AS pick
), rand_id AS (
    SELECT id
    FROM (
        SELECT id, SUM(weight) OVER (ORDER BY id) AS scaled_weight, pick
        FROM weights CROSS JOIN rand_pick
    ) AS q
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

-- name: GetCell :one
SELECT id, action, max_no_ops, initial, state
FROM cells
WHERE id = $1;

-- name: InsertCell :one
INSERT INTO cells (
    hash, hash_input, action, max_no_ops, initial, section, state
) VALUES (
    $1, $2, $3, $4, $5, $6, $7
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

-- name: SetCellInvalid :exec
UPDATE cells
SET invalid = TRUE
where id = $1;

-- name: DeleteOldCellScores :exec
WITH ranked_scores AS (
    SELECT 
        id,
        cell_id,
        ROW_NUMBER() OVER (PARTITION BY cell_id ORDER BY id DESC) AS rn
    FROM cell_scores
)
DELETE FROM cell_scores
WHERE id IN (
    SELECT id
    FROM ranked_scores
    WHERE rn > 10
);
