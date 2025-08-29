### BF for TASK 1 ####
# ==== CONFIG ====
gold_path <- "TASK1_BF/TASK1_test_for_internal_use.csv"
submission_paths <- c(
  "TASK1_BF/TASK1_test_predictions.csv",
  "TASK1_BF/TASK1_test_10model_ensemble_predictions.csv",
  "TASK1_BF/TASK1_FINAL_SUBMISSION.csv"
  # add more CSVs here as needed
)


N <- 10000L         # number of bootstraps
sample_frac <- 0.1 # 0.10 is 10% of stimuli per bootstrap (use 1.0 for full-size)
set.seed(98109)

# ==== LIBS ====
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(purrr); library(tibble); library(stringr)
})

# ==== HELPERS ====
cosine_distance <- function(x, y, eps = 1e-12) {
  1 - sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)) + eps)
}
clean_ids <- function(x) toupper(trimws(as.character(x)))
read_tbl <- function(p) readr::read_csv(p, show_col_types = FALSE)

# Align each submission to the gold (by stimulus). Skip if no overlap.
align_to_gold <- function(gold, sub, key = "stimulus", attrs) {
  g_ids <- clean_ids(gold[[key]])
  s_ids <- clean_ids(sub[[key]])
  common <- intersect(g_ids, s_ids)
  if (length(common) == 0) return(NULL)
  # Keep gold order
  gold_f <- gold[g_ids %in% common, ]
  sub_f  <- sub[s_ids %in% common, ]
  sub_f  <- sub_f[match(clean_ids(gold_f[[key]]), clean_ids(sub_f[[key]])), ]
  # Keep the same attribute columns as gold (drop any extras)
  if (!all(attrs %in% names(sub_f))) {
    stop("Submission missing required columns: ", paste(setdiff(attrs, names(sub_f)), collapse = ", "))
  }
  list(
    gold_mat = as.matrix(select(gold_f, all_of(attrs))),
    sub_mat  = as.matrix(select(sub_f,  all_of(attrs))),
    n_rows   = nrow(gold_f),
    common_ids = gold_f[[key]]
  )
}

# Compute Pearson (higher better) and Cosine distance (lower better) over a set of rows
metrics_on_rows <- function(rows, gold_mat, sub_mat) {
  gx <- as.numeric(gold_mat[rows, , drop = FALSE])
  px <- as.numeric(sub_mat[rows, , drop = FALSE])
  pearson <- suppressWarnings(cor(gx, px, method = "pearson"))
  cosdist <- cosine_distance(gx, px)
  c(pearson = pearson, cosine = cosdist)
}

# ==== LOAD GOLD ====
gold <- read_tbl(gold_path)
stopifnot("stimulus" %in% names(gold))
attr_cols <- setdiff(names(gold), "stimulus")  # all attributes in gold

# ==== LOAD SUBMISSIONS & ALIGN ====
if (length(submission_paths) < 2) {
  message("⚠️ Provide at least two participant CSVs in `submission_paths` that share stimuli with the gold.")
}

subs_raw <- set_names(lapply(submission_paths, read_tbl),
                      tools::file_path_sans_ext(basename(submission_paths)))

aligned <- list()
for (nm in names(subs_raw)) {
  sub <- subs_raw[[nm]]
  if (!"stimulus" %in% names(sub)) {
    message("Skipping ", nm, " (no 'stimulus' column).")
    next
  }
  al <- align_to_gold(gold, sub, key = "stimulus", attrs = attr_cols)
  if (is.null(al)) {
    message("Skipping ", nm, " (no overlapping stimulus IDs with gold).")
  } else {
    message("Aligned ", nm, ": ", al$n_rows, " overlapping stimuli.")
    aligned[[nm]] <- al
  }
}

if (length(aligned) < 2) {
  stop("Need at least two aligned submissions with non-empty overlap vs. gold. Check the files listed in `submission_paths`.")
}

# Build matrices keyed by team name (all aligned to the same gold subset).
# If overlaps differ across files, use the intersection of all aligned IDs.
common_ids_all <- Reduce(intersect, lapply(aligned, function(x) clean_ids(x$common_ids)))
if (length(common_ids_all) == 0) stop("Aligned submissions do not share a common set of stimuli. Use a consistent test set.")
# Re-extract matrices restricted to the shared IDs and gold order
gold_ids <- clean_ids(gold$stimulus)
keep_gold <- gold_ids %in% common_ids_all
gold_master <- as.matrix(select(gold[keep_gold, ], all_of(attr_cols)))
order_ids <- clean_ids(gold$stimulus[keep_gold])

team_mats <- list()
for (nm in names(aligned)) {
  sub_ids  <- clean_ids(aligned[[nm]]$common_ids)
  sub_full <- aligned[[nm]]$sub_mat[match(order_ids, sub_ids), , drop = FALSE]
  team_mats[[nm]] <- sub_full
}

team_names <- names(team_mats)
T <- length(team_names)
n <- nrow(gold_master)

# ==== BASELINE RANKING (all shared stimuli) ====
base_metrics <- t(sapply(team_names, function(nm) {
  metrics_on_rows(seq_len(n), gold_master, team_mats[[nm]])
}))
colnames(base_metrics) <- c("pearson","cosine")
rank_pearson <- rank(-base_metrics[, "pearson"], ties.method = "average") # higher = better
rank_cosine  <- rank( base_metrics[, "cosine"],  ties.method = "average") # lower  = better
avg_rank     <- (rank_pearson + rank_cosine) / 2

baseline_tbl <- tibble(
  team = team_names,
  pearson = base_metrics[, "pearson"],
  cosine  = base_metrics[, "cosine"],
  rank_pearson = as.numeric(rank_pearson),
  rank_cosine  = as.numeric(rank_cosine),
  average_rank = as.numeric(avg_rank)
) %>%
  arrange(average_rank, rank_pearson, rank_cosine, desc(pearson), cosine)

print(baseline_tbl, n = nrow(baseline_tbl))
top2 <- baseline_tbl$team[1:2]
cat(sprintf("\nTop-2 by baseline average rank: %s (1st) vs %s (2nd)\n", top2[1], top2[2]))

# ==== BOOTSTRAP ====
B <- max(1L, round(n * sample_frac))
wins <- 0L
ties <- 0L

for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  # ranks per metric
  rP <- rank(-met[, "pearson"], ties.method = "average")
  rC <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
  if (is.na(a1) || is.na(a2)) next
  if (a1 < a2)      wins <- wins + 1L
  else if (a1 == a2) ties <- ties + 1L  # ties counted as 0.5
}

p_win <- (wins + 0.5 * ties) / N           # "your" BF if you define BF = p_win
BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)  # optional odds form

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))

# Save a small summary
readr::write_csv(baseline_tbl, "TASK1_baseline_ranking.csv")
cat("\nWrote: TASK1_baseline_ranking.csv\n")

### BF for TASK 2 ####
# ==== CONFIG ====
gold_path <- "TASK2_BF/TASK2_testDataset_for_internal_use.csv"
submission_paths <- c(
  "TASK2_BF/task2_test_ensemble.csv",
  "TASK2_BF/predictions_quantile_normalized_dynamicadd.csv",
  "TASK2_BF/predictions_quantile_normalized.add.csv",
  "TASK2_BF/TASK2_FINAL_SUBMISSION.csv"
  # add more CSVs here as needed
)


N <- 10000L         # number of bootstraps
sample_frac <- 0.1  # 0.1 is 10% of stimuli per bootstrap (use 1.0 for full-size)
set.seed(98109)

# ==== LIBS ====
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(purrr); library(tibble); library(stringr)
})

# ==== HELPERS ====
cosine_distance <- function(x, y, eps = 1e-12) {
  1 - sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)) + eps)
}
clean_ids <- function(x) toupper(trimws(as.character(x)))
read_tbl <- function(p) readr::read_csv(p, show_col_types = FALSE)

# Align each submission to the gold (by stimulus). Skip if no overlap.
align_to_gold <- function(gold, sub, key = "stimulus", attrs) {
  g_ids <- clean_ids(gold[[key]])
  s_ids <- clean_ids(sub[[key]])
  common <- intersect(g_ids, s_ids)
  if (length(common) == 0) return(NULL)
  # Keep gold order
  gold_f <- gold[g_ids %in% common, ]
  sub_f  <- sub[s_ids %in% common, ]
  sub_f  <- sub_f[match(clean_ids(gold_f[[key]]), clean_ids(sub_f[[key]])), ]
  # Keep the same attribute columns as gold (drop any extras)
  if (!all(attrs %in% names(sub_f))) {
    stop("Submission missing required columns: ", paste(setdiff(attrs, names(sub_f)), collapse = ", "))
  }
  list(
    gold_mat = as.matrix(select(gold_f, all_of(attrs))),
    sub_mat  = as.matrix(select(sub_f,  all_of(attrs))),
    n_rows   = nrow(gold_f),
    common_ids = gold_f[[key]]
  )
}

# Compute Pearson (higher better) and Cosine distance (lower better) over a set of rows
metrics_on_rows <- function(rows, gold_mat, sub_mat) {
  gx <- as.numeric(gold_mat[rows, , drop = FALSE])
  px <- as.numeric(sub_mat[rows, , drop = FALSE])
  pearson <- suppressWarnings(cor(gx, px, method = "pearson"))
  cosdist <- cosine_distance(gx, px)
  c(pearson = pearson, cosine = cosdist)
}

# ==== LOAD GOLD ====
gold <- read_tbl(gold_path)
stopifnot("stimulus" %in% names(gold))
attr_cols <- setdiff(names(gold), "stimulus")  # all attributes in gold

# ==== LOAD SUBMISSIONS & ALIGN ====
if (length(submission_paths) < 2) {
  message("⚠️ Provide at least two participant CSVs in `submission_paths` that share stimuli with the gold.")
}

subs_raw <- set_names(lapply(submission_paths, read_tbl),
                      tools::file_path_sans_ext(basename(submission_paths)))

aligned <- list()
for (nm in names(subs_raw)) {
  sub <- subs_raw[[nm]]
  if (!"stimulus" %in% names(sub)) {
    message("Skipping ", nm, " (no 'stimulus' column).")
    next
  }
  al <- align_to_gold(gold, sub, key = "stimulus", attrs = attr_cols)
  if (is.null(al)) {
    message("Skipping ", nm, " (no overlapping stimulus IDs with gold).")
  } else {
    message("Aligned ", nm, ": ", al$n_rows, " overlapping stimuli.")
    aligned[[nm]] <- al
  }
}

if (length(aligned) < 2) {
  stop("Need at least two aligned submissions with non-empty overlap vs. gold. Check the files listed in `submission_paths`.")
}

# Build matrices keyed by team name (all aligned to the same gold subset).
# If overlaps differ across files, use the intersection of all aligned IDs.
common_ids_all <- Reduce(intersect, lapply(aligned, function(x) clean_ids(x$common_ids)))
if (length(common_ids_all) == 0) stop("Aligned submissions do not share a common set of stimuli. Use a consistent test set.")
# Re-extract matrices restricted to the shared IDs and gold order
gold_ids <- clean_ids(gold$stimulus)
keep_gold <- gold_ids %in% common_ids_all
gold_master <- as.matrix(select(gold[keep_gold, ], all_of(attr_cols)))
order_ids <- clean_ids(gold$stimulus[keep_gold])

team_mats <- list()
for (nm in names(aligned)) {
  sub_ids  <- clean_ids(aligned[[nm]]$common_ids)
  sub_full <- aligned[[nm]]$sub_mat[match(order_ids, sub_ids), , drop = FALSE]
  team_mats[[nm]] <- sub_full
}

team_names <- names(team_mats)
T <- length(team_names)
n <- nrow(gold_master)

# ==== BASELINE RANKING (all shared stimuli) ====
base_metrics <- t(sapply(team_names, function(nm) {
  metrics_on_rows(seq_len(n), gold_master, team_mats[[nm]])
}))
colnames(base_metrics) <- c("pearson","cosine")
rank_pearson <- rank(-base_metrics[, "pearson"], ties.method = "average") # higher = better
rank_cosine  <- rank( base_metrics[, "cosine"],  ties.method = "average") # lower  = better
avg_rank     <- (rank_pearson + rank_cosine) / 2

baseline_tbl <- tibble(
  team = team_names,
  pearson = base_metrics[, "pearson"],
  cosine  = base_metrics[, "cosine"],
  rank_pearson = as.numeric(rank_pearson),
  rank_cosine  = as.numeric(rank_cosine),
  average_rank = as.numeric(avg_rank)
) %>%
  arrange(average_rank, rank_pearson, rank_cosine, desc(pearson), cosine)

print(baseline_tbl, n = nrow(baseline_tbl))
top2 <- baseline_tbl$team[1:2]
cat(sprintf("\nTop-2 by baseline average rank: %s (1st) vs %s (2nd)\n", top2[1], top2[2]))

# ==== BOOTSTRAP ====
B <- max(1L, round(n * sample_frac))
wins <- 0L
ties <- 0L

for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  # ranks per metric
  rP <- rank(-met[, "pearson"], ties.method = "average")
  rC <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
  if (is.na(a1) || is.na(a2)) next
  if (a1 < a2)      wins <- wins + 1L
  else if (a1 == a2) ties <- ties + 1L  # ties counted as 0.5
}

p_win <- (wins + 0.5 * ties) / N           # "your" BF if you define BF = p_win
BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)  # optional odds form

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))

# Save a small summary
readr::write_csv(baseline_tbl, "TASK2_baseline_ranking.csv")
cat("\nWrote: TASK2_baseline_ranking.csv\n")


### BF for TASK 1 - Like python script not local ####
# ==== CONFIG ====
gold_path <- "TASK1_BF/TASK1_test_for_internal_use.csv"
submission_paths <- c(
  "TASK1_BF/TASK1_test_predictions.csv",
  "TASK1_BF/TASK1_test_10model_ensemble_predictions.csv",
  "TASK1_BF/TASK1_FINAL_SUBMISSION.csv"
)

N <- 10000L      # number of bootstraps
sample_frac <- 1.0  # full-size bootstrap; set 0.10 for 10%
set.seed(98109)

# ==== LIBS ====
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(purrr); library(tibble); library(stringr)
})

# ==== HELPERS (python-style) ====
cosine_distance <- function(x, y, eps = 1e-12) {
  1 - sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)) + eps)
}
clean_ids <- function(x) toupper(trimws(as.character(x)))
read_tbl  <- function(p) readr::read_csv(p, show_col_types = FALSE)

align_to_gold <- function(gold, sub, key = "stimulus", target_cols) {
  g_ids <- clean_ids(gold[[key]])
  s_ids <- clean_ids(sub[[key]])
  common <- intersect(g_ids, s_ids)
  if (length(common) == 0) return(NULL)
  
  gold_f <- gold[g_ids %in% common, ]
  sub_f  <- sub[s_ids %in% common, ]
  sub_f  <- sub_f[match(clean_ids(gold_f[[key]]), clean_ids(sub_f[[key]])), ]
  
  if (!all(target_cols %in% names(sub_f))) {
    stop("Submission missing required columns: ",
         paste(setdiff(target_cols, names(sub_f)), collapse = ", "))
  }
  
  G <- as.matrix(dplyr::select(gold_f, dplyr::all_of(target_cols)))
  P <- as.matrix(dplyr::select(sub_f,  dplyr::all_of(target_cols)))
  storage.mode(G) <- "double"; storage.mode(P) <- "double"
  
  list(
    gold_mat   = G,
    sub_mat    = P,
    n_rows     = nrow(G),
    common_ids = gold_f[[key]]
  )
}

# Python-style: per-row metrics, averaged across rows
metrics_on_rows <- function(rows, gold_mat, sub_mat) {
  P <- sapply(rows, function(i) {
    suppressWarnings(cor(gold_mat[i, ], sub_mat[i, ], method = "pearson"))
  })
  C <- sapply(rows, function(i) {
    cosine_distance(gold_mat[i, ], sub_mat[i, ])
  })
  c(pearson = mean(P, na.rm = TRUE),
    cosine  = mean(C, na.rm = TRUE))
}

# ==== LOAD GOLD ====
gold <- read_tbl(gold_path)
stopifnot("stimulus" %in% names(gold))

# Use the gold's numeric columns (except 'stimulus')—or hard-code the official target set.
target_cols <- names(gold)[sapply(gold, is.numeric)]
target_cols <- setdiff(target_cols, "stimulus")
# Example if official scorer used a single column:
# target_cols <- c("Experimental_Values")

# ==== LOAD SUBMISSIONS & ALIGN ====
subs_raw <- setNames(lapply(submission_paths, read_tbl),
                     tools::file_path_sans_ext(basename(submission_paths)))

aligned <- list()
for (nm in names(subs_raw)) {
  sub <- subs_raw[[nm]]
  if (!"stimulus" %in% names(sub)) {
    message("Skipping ", nm, " (no 'stimulus' column).")
    next
  }
  al <- align_to_gold(gold, sub, key = "stimulus", target_cols = target_cols)
  if (is.null(al)) {
    message("Skipping ", nm, " (no overlapping stimulus IDs with gold).")
  } else {
    message("Aligned ", nm, ": ", al$n_rows, " overlapping stimuli.")
    aligned[[nm]] <- al
  }
}

stopifnot(length(aligned) >= 2)
common_ids_all <- Reduce(intersect, lapply(aligned, function(x) clean_ids(x$common_ids)))
stopifnot(length(common_ids_all) > 0)

# Rebuild master gold and submission matrices on the shared IDs (gold order)
gold_ids <- clean_ids(gold$stimulus)
keep_gold <- gold_ids %in% common_ids_all
order_ids <- clean_ids(gold$stimulus[keep_gold])
gold_master <- as.matrix(dplyr::select(gold[keep_gold, ], dplyr::all_of(target_cols)))
storage.mode(gold_master) <- "double"

team_mats <- list()
for (nm in names(aligned)) {
  sub_ids  <- clean_ids(aligned[[nm]]$common_ids)
  sub_full <- aligned[[nm]]$sub_mat[match(order_ids, sub_ids), , drop = FALSE]
  storage.mode(sub_full) <- "double"
  team_mats[[nm]] <- sub_full
}

team_names <- names(team_mats)
n <- nrow(gold_master)

# ==== BASELINE (no bootstrap) ====
base_metrics <- t(sapply(team_names, function(nm) {
  metrics_on_rows(seq_len(n), gold_master, team_mats[[nm]])
}))
colnames(base_metrics) <- c("pearson","cosine")
rank_pearson <- rank(-base_metrics[, "pearson"], ties.method = "average") # higher = better
rank_cosine  <- rank( base_metrics[, "cosine"],  ties.method = "average") # lower  = better
avg_rank     <- (rank_pearson + rank_cosine) / 2

baseline_tbl <- tibble::tibble(
  team = team_names,
  pearson = base_metrics[, "pearson"],
  cosine  = base_metrics[, "cosine"],
  rank_pearson = as.numeric(rank_pearson),
  rank_cosine  = as.numeric(rank_cosine),
  average_rank = as.numeric(avg_rank)
) %>%
  arrange(average_rank, rank_pearson, rank_cosine, desc(pearson), cosine)

print(baseline_tbl, n = nrow(baseline_tbl))

# ==== BOOTSTRAP p_win / odds ====
top2 <- baseline_tbl$team[1:2]
B <- max(1L, round(n * sample_frac))
wins <- 0L; ties <- 0L

for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP <- rank(-met[, "pearson"], ties.method = "average")
  rC <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
  if (is.na(a1) || is.na(a2)) next
  if (a1 < a2)      wins <- wins + 1L
  else if (a1 == a2) ties <- ties + 1L
}

p_win  <- (wins + 0.5 * ties) / N
BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))

### BF for TASK 2 - Like python script not local ####
# ==== CONFIG ====
gold_path <- "TASK2_BF/TASK2_testDataset_for_internal_use.csv"
submission_paths <- c(
  "TASK2_BF/task2_test_ensemble.csv",
  "TASK2_BF/predictions_quantile_normalized_dynamicadd.csv",
  "TASK2_BF/predictions_quantile_normalized.add.csv",
  "TASK2_BF/TASK2_FINAL_SUBMISSION.csv"
  # add more CSVs here as needed
)

N <- 10000L      # number of bootstraps
sample_frac <- 1.0  # full-size bootstrap; set 0.10 for 10%
set.seed(98109)

# ==== LIBS ====
suppressPackageStartupMessages({
  library(readr); library(dplyr); library(purrr); library(tibble); library(stringr)
})

# ==== HELPERS (python-style) ====
cosine_distance <- function(x, y, eps = 1e-12) {
  1 - sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)) + eps)
}
clean_ids <- function(x) toupper(trimws(as.character(x)))
read_tbl  <- function(p) readr::read_csv(p, show_col_types = FALSE)

align_to_gold <- function(gold, sub, key = "stimulus", target_cols) {
  g_ids <- clean_ids(gold[[key]])
  s_ids <- clean_ids(sub[[key]])
  common <- intersect(g_ids, s_ids)
  if (length(common) == 0) return(NULL)
  
  gold_f <- gold[g_ids %in% common, ]
  sub_f  <- sub[s_ids %in% common, ]
  sub_f  <- sub_f[match(clean_ids(gold_f[[key]]), clean_ids(sub_f[[key]])), ]
  
  if (!all(target_cols %in% names(sub_f))) {
    stop("Submission missing required columns: ",
         paste(setdiff(target_cols, names(sub_f)), collapse = ", "))
  }
  
  G <- as.matrix(dplyr::select(gold_f, dplyr::all_of(target_cols)))
  P <- as.matrix(dplyr::select(sub_f,  dplyr::all_of(target_cols)))
  storage.mode(G) <- "double"; storage.mode(P) <- "double"
  
  list(
    gold_mat   = G,
    sub_mat    = P,
    n_rows     = nrow(G),
    common_ids = gold_f[[key]]
  )
}

# Python-style: per-row metrics, averaged across rows
metrics_on_rows <- function(rows, gold_mat, sub_mat) {
  P <- sapply(rows, function(i) {
    suppressWarnings(cor(gold_mat[i, ], sub_mat[i, ], method = "pearson"))
  })
  C <- sapply(rows, function(i) {
    cosine_distance(gold_mat[i, ], sub_mat[i, ])
  })
  c(pearson = mean(P, na.rm = TRUE),
    cosine  = mean(C, na.rm = TRUE))
}

# ==== LOAD GOLD ====
gold <- read_tbl(gold_path)
stopifnot("stimulus" %in% names(gold))

# Use the gold's numeric columns (except 'stimulus')—or hard-code the official target set.
target_cols <- names(gold)[sapply(gold, is.numeric)]
target_cols <- setdiff(target_cols, "stimulus")
# Example if official scorer used a single column:
# target_cols <- c("Experimental_Values")

# ==== LOAD SUBMISSIONS & ALIGN ====
subs_raw <- setNames(lapply(submission_paths, read_tbl),
                     tools::file_path_sans_ext(basename(submission_paths)))

aligned <- list()
for (nm in names(subs_raw)) {
  sub <- subs_raw[[nm]]
  if (!"stimulus" %in% names(sub)) {
    message("Skipping ", nm, " (no 'stimulus' column).")
    next
  }
  al <- align_to_gold(gold, sub, key = "stimulus", target_cols = target_cols)
  if (is.null(al)) {
    message("Skipping ", nm, " (no overlapping stimulus IDs with gold).")
  } else {
    message("Aligned ", nm, ": ", al$n_rows, " overlapping stimuli.")
    aligned[[nm]] <- al
  }
}

stopifnot(length(aligned) >= 2)
common_ids_all <- Reduce(intersect, lapply(aligned, function(x) clean_ids(x$common_ids)))
stopifnot(length(common_ids_all) > 0)

# Rebuild master gold and submission matrices on the shared IDs (gold order)
gold_ids <- clean_ids(gold$stimulus)
keep_gold <- gold_ids %in% common_ids_all
order_ids <- clean_ids(gold$stimulus[keep_gold])
gold_master <- as.matrix(dplyr::select(gold[keep_gold, ], dplyr::all_of(target_cols)))
storage.mode(gold_master) <- "double"

team_mats <- list()
for (nm in names(aligned)) {
  sub_ids  <- clean_ids(aligned[[nm]]$common_ids)
  sub_full <- aligned[[nm]]$sub_mat[match(order_ids, sub_ids), , drop = FALSE]
  storage.mode(sub_full) <- "double"
  team_mats[[nm]] <- sub_full
}

team_names <- names(team_mats)
n <- nrow(gold_master)

# ==== BASELINE (no bootstrap) ====
base_metrics <- t(sapply(team_names, function(nm) {
  metrics_on_rows(seq_len(n), gold_master, team_mats[[nm]])
}))
colnames(base_metrics) <- c("pearson","cosine")
rank_pearson <- rank(-base_metrics[, "pearson"], ties.method = "average") # higher = better
rank_cosine  <- rank( base_metrics[, "cosine"],  ties.method = "average") # lower  = better
avg_rank     <- (rank_pearson + rank_cosine) / 2

baseline_tbl <- tibble::tibble(
  team = team_names,
  pearson = base_metrics[, "pearson"],
  cosine  = base_metrics[, "cosine"],
  rank_pearson = as.numeric(rank_pearson),
  rank_cosine  = as.numeric(rank_cosine),
  average_rank = as.numeric(avg_rank)
) %>%
  arrange(average_rank, rank_pearson, rank_cosine, desc(pearson), cosine)

print(baseline_tbl, n = nrow(baseline_tbl))

# ==== BOOTSTRAP p_win / odds ====
top2 <- baseline_tbl$team[1:2]
B <- max(1L, round(n * sample_frac))
wins <- 0L; ties <- 0L

for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP <- rank(-met[, "pearson"], ties.method = "average")
  rC <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
  if (is.na(a1) || is.na(a2)) next
  if (a1 < a2)      wins <- wins + 1L
  else if (a1 == a2) ties <- ties + 1L
}

p_win  <- (wins + 0.5 * ties) / N
BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))
