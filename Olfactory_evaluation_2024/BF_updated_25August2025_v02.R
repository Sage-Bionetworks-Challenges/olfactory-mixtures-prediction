## updated script for Person and Cosine calcualtion. Previous script did a global person and cosine ## assessment. It flattens all selected values into a single vector per submission and computes one ##global Pearson and one global cosine distanceThis script computes Pearson and cosine per stimulus ##(row) across features, then ## ## averages those per-row values. That’s what the final-round CSVs ## reflect.

###### BF for TASK 1 - Like python script not local ####
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
# top2 <- baseline_tbl$team[1:2]
# B <- max(1L, round(n * sample_frac))
# wins <- 0L; ties <- 0L
# 
# for (b in seq_len(N)) {
#   rows <- sample.int(n, size = B, replace = TRUE)
#   met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
#   rP <- rank(-met[, "pearson"], ties.method = "average")
#   rC <- rank( met[, "cosine"],  ties.method = "average")
#   avg <- (rP + rC) / 2
#   a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
#   if (is.na(a1) || is.na(a2)) next
#   if (a1 < a2)      wins <- wins + 1L
#   else if (a1 == a2) ties <- ties + 1L
# }
# 
# p_win  <- (wins + 0.5 * ties) / N
# BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)
# 
# cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
# cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
# cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))

# ==== BOOTSTRAP p_win / odds ====
# top2 <- baseline_tbl$team[1:2]
# B <- max(1L, round(n * sample_frac))
# wins <- 0L; ties <- 0L
# 
# for (b in seq_len(N)) {
#   rows <- sample.int(n, size = B, replace = TRUE)
#   met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
#   rP <- rank(-met[, "pearson"], ties.method = "average")
#   rC <- rank( met[, "cosine"],  ties.method = "average")
#   avg <- (rP + rC) / 2
#   a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
#   if (is.na(a1) || is.na(a2)) next
#   if (a1 < a2)      wins <- wins + 1L
#   else if (a1 == a2) ties <- ties + 1L
# }
# 
# p_win  <- (wins + 0.5 * ties) / N
# BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)

# ==== BOOTSTRAP p_win / odds — extended (1 vs 2) and (2 vs 3) ====
B <- max(1L, round(n * sample_frac))

# Build the comparison pairs from the baseline order
pairs <- list(
  c(baseline_tbl$team[1], baseline_tbl$team[2])   # 1 vs 2
)
if (nrow(baseline_tbl) >= 3) {
  pairs <- c(pairs, list(c(baseline_tbl$team[2], baseline_tbl$team[3])))  # 2 vs 3
}

pair_names <- vapply(pairs, function(p) paste0(p[1], " vs ", p[2]), character(1))
wins <- setNames(integer(length(pairs)), pair_names)
ties <- setNames(integer(length(pairs)), pair_names)

set.seed(98109)
for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  
  # metrics for all teams on these rows
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP  <- rank(-met[, "pearson"], ties.method = "average")
  rC  <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  names(avg) <- rownames(met)
  
  # update every comparison on this draw
  for (i in seq_along(pairs)) {
    a <- avg[pairs[[i]][1]]
    b2 <- avg[pairs[[i]][2]]
    if (is.na(a) || is.na(b2)) next
    if (a < b2)       wins[i] <- wins[i] + 1L
    else if (a == b2) ties[i] <- ties[i] + 1L
  }
}

# summarize
res <- tibble::tibble(
  comparison = names(wins),
  wins = as.integer(wins),
  ties = as.integer(ties),
  N = N,
  p_win = (wins + 0.5 * ties) / N,
  BF_odds = (p_win + 1e-8) / (1 - p_win + 1e-8),
  rows_per_draw = B,
  sample_frac = sample_frac
)

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n",
            N, B, sample_frac))
print(res, n = nrow(res))

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))

team2 <- baseline_tbl$team[2]
team3 <- baseline_tbl$team[3]

res_2v3 <- dplyr::filter(res, comparison == paste0(team2, " vs ", team3))
print(res_2v3)

cat(sprintf("%s vs %s  ->  p_win = %.4f, BF_odds = %.3f\n",
            team2, team3, res_2v3$p_win, res_2v3$BF_odds))

cat(sprintf("%s vs %s  ->  p_win = %.4f, BF_odds = %.3f\n",
            team2, team3, res_2v3$p_win, res_2v3$BF_odds))

knitr::kable(res[, c("comparison","p_win","BF_odds","rows_per_draw","sample_frac")],
             digits = 3)



# ==== THREE-WAY TIE CHECK (pairwise among top-3) ====

# --- settings ---
B          <- max(1L, round(n * sample_frac))   # rows per draw, as in your script
N_draws    <- N                                 # same number of bootstraps
tie_rule   <- "BF"                               # "BF" or "pwin"
bf_cutoff  <- 3                                  # BF tie window is [1/3, 3]
pwin_eps   <- 0.05                               # p_win tie window is [0.5 - eps, 0.5 + eps]

# build the top-3 list (if fewer than 3, it will do whatever is available)
top3 <- head(baseline_tbl$team, 3)
if (length(top3) < 2L) stop("Need at least two teams to compare.")

# all unordered pairs among the chosen teams
pairs <- combn(top3, 2, simplify = FALSE)

pair_key <- function(a,b) paste0(a, " vs ", b)

# counters
wins <- setNames(integer(length(pairs)), sapply(pairs, \(p) pair_key(p[1], p[2])))
ties <- setNames(integer(length(pairs)), names(wins))

set.seed(98109)
for (b in seq_len(N_draws)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  
  # metrics for all teams on these rows
  met <- t(sapply(names(team_mats), function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP  <- rank(-met[, "pearson"], ties.method = "average")
  rC  <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  names(avg) <- rownames(met)
  
  # update each pair's win/tie wrt ordering as written in the pair (first beats second)
  for (i in seq_along(pairs)) {
    a <- avg[pairs[[i]][1]]
    b2 <- avg[pairs[[i]][2]]
    if (is.na(a) || is.na(b2)) next
    if (a < b2)       wins[i] <- wins[i] + 1L
    else if (a == b2) ties[i] <- ties[i] + 1L
  }
}

pair_res <- tibble::tibble(
  comparison     = names(wins),
  team_A         = sapply(pairs, `[`, 1),
  team_B         = sapply(pairs, `[`, 2),
  wins           = as.integer(wins),
  ties           = as.integer(ties),
  N              = N_draws,
  p_win          = (wins + 0.5 * ties) / N_draws,                 # Pr(A beats B)
  BF_odds        = (p_win + 1e-8) / (1 - p_win + 1e-8),           # odds form
  rows_per_draw  = B,
  sample_frac    = sample_frac
)

# tie decision per pair
pair_res <- pair_res %>%
  mutate(
    tie_by_BF   = BF_odds >= 1 / bf_cutoff & BF_odds <= bf_cutoff,
    tie_by_pwin = p_win >= (0.5 - pwin_eps) & p_win <= (0.5 + pwin_eps),
    tie_flag    = ifelse(tie_rule == "BF", tie_by_BF, tie_by_pwin)
  )

# is there a three-way tie among top-3?
three_way_tie <-
  (length(top3) >= 3) &&
  all(pair_res$tie_flag[pair_res$team_A %in% top3 & pair_res$team_B %in% top3])

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n",
            N_draws, B, sample_frac))

print(pair_res %>%
        select(comparison, p_win, BF_odds, rows_per_draw, sample_frac,
               tie_by_BF, tie_by_pwin, tie_flag),
      n = nrow(pair_res))

cat("\nTie rule used: ", if (tie_rule == "BF")
  sprintf("BF window [1/%.1f, %.1f]", bf_cutoff, bf_cutoff)
  else
    sprintf("p_win window [%.3f, %.3f]", 0.5 - pwin_eps, 0.5 + pwin_eps),
  "\n", sep="")

if (length(top3) >= 3) {
  cat(sprintf("Top-3: %s\n", paste(top3, collapse = " | ")))
  cat(sprintf("Three-way tie among top-3? %s\n", ifelse(three_way_tie, "YES", "NO")))
} else {
  cat("Fewer than 3 teams available; three-way tie not applicable.\n")
}

# --- assumes you've already defined:
# gold_master, team_mats (named list of matrices), team_names, baseline_tbl, N, sample_frac

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(stringr); library(forcats)
})

# helper reused from your script
cosine_distance <- function(x, y, eps = 1e-12) {
  1 - sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)) + eps)
}
metrics_on_rows <- function(rows, gold_mat, sub_mat) {
  # per-row Pearson & cosine, then average across rows
  P <- sapply(rows, function(i) suppressWarnings(cor(gold_mat[i, ], sub_mat[i, ], method = "pearson")))
  C <- sapply(rows, function(i) cosine_distance(gold_mat[i, ], sub_mat[i, ]))
  c(pearson = mean(P, na.rm = TRUE), cosine = mean(C, na.rm = TRUE))
}

set.seed(98109)
n <- nrow(gold_master)
B <- max(1L, round(n * sample_frac))

# --- 1) Collect per-bootstrap average ranks for each team
boot_df <- map_dfr(seq_len(N), function(bi){
  rows <- sample.int(n, size = B, replace = TRUE)
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP <- rank(-met[, "pearson"], ties.method = "average")
  rC <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC)/2
  tibble(boot = bi, team = names(avg), avg_rank = as.numeric(avg))
})

# keep the top-3 teams by baseline (if you want exactly top-3)
top3 <- baseline_tbl %>% arrange(average_rank, rank_pearson, rank_cosine) %>% slice(1:3) %>% pull(team)
boot_top3 <- boot_df %>% filter(team %in% top3)

# --- 2) Pairwise BF odds among the top-3 (all three pairings)
pair_grid <- t(combn(top3, 2))
pair_names <- paste(pair_grid[,1], "vs", pair_grid[,2])

bf_tbl <- map2_dfr(as.data.frame(pair_grid)[,1], as.data.frame(pair_grid)[,2], function(a, b){
  comp <- boot_top3 %>%
    filter(team %in% c(a,b)) %>%
    pivot_wider(names_from = team, values_from = avg_rank)
  
  # wins for "a" are cases where a has lower (better) avg_rank
  wins_a <- sum(comp[[a]] <  comp[[b]], na.rm = TRUE)
  ties   <- sum(comp[[a]] == comp[[b]], na.rm = TRUE)
  N_eff  <- nrow(comp)
  
  p_win  <- (wins_a + 0.5 * ties) / N_eff
  BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)
  
  tibble(
    comparison = paste(a, "vs", b),
    p_win = p_win,
    BF_odds = BF_odds,
    N = N_eff,
    rows_per_draw = B,
    sample_frac = sample_frac,
    tie_by_BF   = BF_odds >= (1/3) & BF_odds <= 3
  )
})

print(bf_tbl, n = nrow(bf_tbl))

# --- 3a) Plot distribution of bootstrap average ranks (lower is better)
p_dist <- boot_top3 %>%
  mutate(team = fct_relevel(team, top3)) %>%
  ggplot(aes(x = team, y = avg_rank)) +
  geom_boxplot(width = 0.6, outlier.alpha = 0.25) +
  coord_flip() +
  labs(title = "Bootstrap distributions of average rank (Pearson+Cosine)",
       x = "Team", y = "Average rank across metrics (lower = better)",
       subtitle = sprintf("N=%d bootstraps, rows/draw=%d, sample_frac=%.2f", N, B, sample_frac)) +
  theme_bw()

# --- 3b) Plot Bayes Factor odds with tie band (1/3–3), log scale helps
p_bf <- bf_tbl %>%
  ggplot(aes(x = fct_reorder(comparison, BF_odds), y = BF_odds)) +
  geom_hline(yintercept = c(1/3, 3), linetype = 2) +
  geom_bar(stat = "identity", width = 0.65) +
  scale_y_log10() +
  coord_flip() +
  labs(title = "Pairwise Bayes Factor (odds) among top-3",
       subtitle = "Dashed lines: tie band (1/3 to 3). Log scale.",
       x = "Comparison", y = "BF odds (log scale)") +
  theme_bw()

print(p_dist)
print(p_bf)


###### BF for TASK 2 - Like python script not local ####
# ==== CONFIG ====
gold_path <- "TASK2_BF/TASK2_testDataset_for_internal_use.csv"
submission_paths <- c(
  "TASK2_BF/task2_test_ensemble.csv",
  #"TASK2_BF/predictions_quantile_normalized.add.csv",
  "TASK2_BF/predictions_quantile_normalized_dynamicadd.csv", ## latest submission
  "TASK2_BF/TASK2_FINAL_SUBMISSION.csv",
  "TASK2_BF/TASK2_Test_Set_Submissions_Temperature_Tuned.csv"
  # add more CSVs here as needed   
)

N <- 10000L      # number of bootstraps
sample_frac <- 0.1  # full-size bootstrap; set 0.10 for 10%
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
# top2 <- baseline_tbl$team[1:2]
# B <- max(1L, round(n * sample_frac))
# wins <- 0L; ties <- 0L
# 
# for (b in seq_len(N)) {
#   rows <- sample.int(n, size = B, replace = TRUE)
#   met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
#   rP <- rank(-met[, "pearson"], ties.method = "average")
#   rC <- rank( met[, "cosine"],  ties.method = "average")
#   avg <- (rP + rC) / 2
#   a1 <- avg[top2[1]]; a2 <- avg[top2[2]]
#   if (is.na(a1) || is.na(a2)) next
#   if (a1 < a2)      wins <- wins + 1L
#   else if (a1 == a2) ties <- ties + 1L
# }
# 
# p_win  <- (wins + 0.5 * ties) / N
# BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)

# ==== BOOTSTRAP p_win / odds — extended (1 vs 2) and (2 vs 3) ====
B <- max(1L, round(n * sample_frac))

# Build the comparison pairs from the baseline order
pairs <- list(
  c(baseline_tbl$team[1], baseline_tbl$team[2])   # 1 vs 2
)
if (nrow(baseline_tbl) >= 3) {
  pairs <- c(pairs, list(c(baseline_tbl$team[2], baseline_tbl$team[3])))  # 2 vs 3
}

pair_names <- vapply(pairs, function(p) paste0(p[1], " vs ", p[2]), character(1))
wins <- setNames(integer(length(pairs)), pair_names)
ties <- setNames(integer(length(pairs)), pair_names)

set.seed(98109)
for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  
  # metrics for all teams on these rows
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP  <- rank(-met[, "pearson"], ties.method = "average")
  rC  <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  names(avg) <- rownames(met)
  
  # update every comparison on this draw
  for (i in seq_along(pairs)) {
    a <- avg[pairs[[i]][1]]
    b2 <- avg[pairs[[i]][2]]
    if (is.na(a) || is.na(b2)) next
    if (a < b2)       wins[i] <- wins[i] + 1L
    else if (a == b2) ties[i] <- ties[i] + 1L
  }
}

# summarize
res <- tibble::tibble(
  comparison = names(wins),
  wins = as.integer(wins),
  ties = as.integer(ties),
  N = N,
  p_win = (wins + 0.5 * ties) / N,
  BF_odds = (p_win + 1e-8) / (1 - p_win + 1e-8),
  rows_per_draw = B,
  sample_frac = sample_frac
)

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n",
            N, B, sample_frac))
print(res, n = nrow(res))

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n", N, B, sample_frac))
cat(sprintf("p_win (rank-1 beats rank-2): %.4f\n", p_win))
cat(sprintf("BF (odds p/(1-p)): %.3f\n", BF_odds))

team2 <- baseline_tbl$team[2]
team3 <- baseline_tbl$team[3]

res_2v3 <- dplyr::filter(res, comparison == paste0(team2, " vs ", team3))
print(res_2v3)

cat(sprintf("%s vs %s  ->  p_win = %.4f, BF_odds = %.3f\n",
            team2, team3, res_2v3$p_win, res_2v3$BF_odds))

cat(sprintf("%s vs %s  ->  p_win = %.4f, BF_odds = %.3f\n",
            team2, team3, res_2v3$p_win, res_2v3$BF_odds))

knitr::kable(res[, c("comparison","p_win","BF_odds","rows_per_draw","sample_frac")],
             digits = 3)


# ==== THREE-WAY TIE CHECK (pairwise among top-3) ====

# --- settings ---
B          <- max(1L, round(n * sample_frac))   # rows per draw, as in your script
N_draws    <- N                                 # same number of bootstraps
tie_rule   <- "BF"                               # "BF" or "pwin"
bf_cutoff  <- 3                                  # BF tie window is [1/3, 3]
pwin_eps   <- 0.05                               # p_win tie window is [0.5 - eps, 0.5 + eps]

# build the top-3 list (if fewer than 3, it will do whatever is available)
top3 <- head(baseline_tbl$team, 3)
if (length(top3) < 2L) stop("Need at least two teams to compare.")

# all unordered pairs among the chosen teams
pairs <- combn(top3, 2, simplify = FALSE)

pair_key <- function(a,b) paste0(a, " vs ", b)

# counters
wins <- setNames(integer(length(pairs)), sapply(pairs, \(p) pair_key(p[1], p[2])))
ties <- setNames(integer(length(pairs)), names(wins))

set.seed(98109)
for (b in seq_len(N_draws)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  
  # metrics for all teams on these rows
  met <- t(sapply(names(team_mats), function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP  <- rank(-met[, "pearson"], ties.method = "average")
  rC  <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  names(avg) <- rownames(met)
  
  # update each pair's win/tie wrt ordering as written in the pair (first beats second)
  for (i in seq_along(pairs)) {
    a <- avg[pairs[[i]][1]]
    b2 <- avg[pairs[[i]][2]]
    if (is.na(a) || is.na(b2)) next
    if (a < b2)       wins[i] <- wins[i] + 1L
    else if (a == b2) ties[i] <- ties[i] + 1L
  }
}

pair_res <- tibble::tibble(
  comparison     = names(wins),
  team_A         = sapply(pairs, `[`, 1),
  team_B         = sapply(pairs, `[`, 2),
  wins           = as.integer(wins),
  ties           = as.integer(ties),
  N              = N_draws,
  p_win          = (wins + 0.5 * ties) / N_draws,                 # Pr(A beats B)
  BF_odds        = (p_win + 1e-8) / (1 - p_win + 1e-8),           # odds form
  rows_per_draw  = B,
  sample_frac    = sample_frac
)

# tie decision per pair
pair_res <- pair_res %>%
  mutate(
    tie_by_BF   = BF_odds >= 1 / bf_cutoff & BF_odds <= bf_cutoff,
    tie_by_pwin = p_win >= (0.5 - pwin_eps) & p_win <= (0.5 + pwin_eps),
    tie_flag    = ifelse(tie_rule == "BF", tie_by_BF, tie_by_pwin)
  )

# is there a three-way tie among top-3?
three_way_tie <-
  (length(top3) >= 3) &&
  all(pair_res$tie_flag[pair_res$team_A %in% top3 & pair_res$team_B %in% top3])

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n",
            N_draws, B, sample_frac))

print(pair_res %>%
        select(comparison, p_win, BF_odds, rows_per_draw, sample_frac,
               tie_by_BF, tie_by_pwin, tie_flag),
      n = nrow(pair_res))

cat("\nTie rule used: ", if (tie_rule == "BF")
  sprintf("BF window [1/%.1f, %.1f]", bf_cutoff, bf_cutoff)
  else
    sprintf("p_win window [%.3f, %.3f]", 0.5 - pwin_eps, 0.5 + pwin_eps),
  "\n", sep="")

if (length(top3) >= 3) {
  cat(sprintf("Top-3: %s\n", paste(top3, collapse = " | ")))
  cat(sprintf("Three-way tie among top-3? %s\n", ifelse(three_way_tie, "YES", "NO")))
} else {
  cat("Fewer than 3 teams available; three-way tie not applicable.\n")
}

###
# --- assumes you've already defined:
# gold_master, team_mats (named list of matrices), team_names, baseline_tbl, N, sample_frac

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(ggplot2); library(purrr); library(stringr); library(forcats)
})

# helper reused from your script
cosine_distance <- function(x, y, eps = 1e-12) {
  1 - sum(x * y) / (sqrt(sum(x^2)) * sqrt(sum(y^2)) + eps)
}
metrics_on_rows <- function(rows, gold_mat, sub_mat) {
  # per-row Pearson & cosine, then average across rows
  P <- sapply(rows, function(i) suppressWarnings(cor(gold_mat[i, ], sub_mat[i, ], method = "pearson")))
  C <- sapply(rows, function(i) cosine_distance(gold_mat[i, ], sub_mat[i, ]))
  c(pearson = mean(P, na.rm = TRUE), cosine = mean(C, na.rm = TRUE))
}

set.seed(98109)
n <- nrow(gold_master)
B <- max(1L, round(n * sample_frac))

# --- 1) Collect per-bootstrap average ranks for each team
boot_df <- map_dfr(seq_len(N), function(bi){
  rows <- sample.int(n, size = B, replace = TRUE)
  met <- t(sapply(team_names, function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP <- rank(-met[, "pearson"], ties.method = "average")
  rC <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC)/2
  tibble(boot = bi, team = names(avg), avg_rank = as.numeric(avg))
})

# keep the top-3 teams by baseline (if you want exactly top-3)
top3 <- baseline_tbl %>% arrange(average_rank, rank_pearson, rank_cosine) %>% slice(1:3) %>% pull(team)
boot_top3 <- boot_df %>% filter(team %in% top3)

# --- 2) Pairwise BF odds among the top-3 (all three pairings)
pair_grid <- t(combn(top3, 2))
pair_names <- paste(pair_grid[,1], "vs", pair_grid[,2])

bf_tbl <- map2_dfr(as.data.frame(pair_grid)[,1], as.data.frame(pair_grid)[,2], function(a, b){
  comp <- boot_top3 %>%
    filter(team %in% c(a,b)) %>%
    pivot_wider(names_from = team, values_from = avg_rank)
  
  # wins for "a" are cases where a has lower (better) avg_rank
  wins_a <- sum(comp[[a]] <  comp[[b]], na.rm = TRUE)
  ties   <- sum(comp[[a]] == comp[[b]], na.rm = TRUE)
  N_eff  <- nrow(comp)
  
  p_win  <- (wins_a + 0.5 * ties) / N_eff
  BF_odds <- (p_win + 1e-8) / (1 - p_win + 1e-8)
  
  tibble(
    comparison = paste(a, "vs", b),
    p_win = p_win,
    BF_odds = BF_odds,
    N = N_eff,
    rows_per_draw = B,
    sample_frac = sample_frac,
    tie_by_BF   = BF_odds >= (1/3) & BF_odds <= 3
  )
})

print(bf_tbl, n = nrow(bf_tbl))

# --- 3a) Plot distribution of bootstrap average ranks (lower is better)
p_dist <- boot_top3 %>%
  mutate(team = fct_relevel(team, top3)) %>%
  ggplot(aes(x = team, y = avg_rank)) +
  geom_boxplot(width = 0.6, outlier.alpha = 0.25) +
  coord_flip() +
  labs(title = "Bootstrap distributions of average rank (Pearson+Cosine)",
       x = "Team", y = "Average rank across metrics (lower = better)",
       subtitle = sprintf("N=%d bootstraps, rows/draw=%d, sample_frac=%.2f", N, B, sample_frac)) +
  theme_bw()

# --- 3b) Plot Bayes Factor odds with tie band (1/3–3), log scale helps
p_bf <- bf_tbl %>%
  ggplot(aes(x = fct_reorder(comparison, BF_odds), y = BF_odds)) +
  geom_hline(yintercept = c(1/3, 3), linetype = 2) +
  geom_bar(stat = "identity", width = 0.65) +
  scale_y_log10() +
  coord_flip() +
  labs(title = "Pairwise Bayes Factor (odds) among top-3",
       subtitle = "Dashed lines: tie band (1/3 to 3). Log scale.",
       x = "Comparison", y = "BF odds (log scale)") +
  theme_bw()

print(p_dist)
print(p_bf)

# ---- 4-way tie bootstrap among top 4 teams ----
B <- max(1L, round(n * sample_frac))

# pick the teams to test (top 4 by baseline average_rank, or fewer if not available)
teams4 <- baseline_tbl$team[seq_len(min(4, nrow(baseline_tbl)))]
stopifnot(length(teams4) >= 2)

# all pairwise comparisons within the chosen set
pair_list <- combn(teams4, 2, simplify = FALSE)
pair_names <- vapply(pair_list, function(p) paste0(p[1], " vs ", p[2]), character(1))
wins <- setNames(integer(length(pair_list)), pair_names)
ties <- setNames(integer(length(pair_list)), pair_names)

set.seed(98109)
for (b in seq_len(N)) {
  rows <- sample.int(n, size = B, replace = TRUE)
  # metrics for all teams on this draw
  met <- t(sapply(names(team_mats), function(nm) metrics_on_rows(rows, gold_master, team_mats[[nm]])))
  rP  <- rank(-met[, "pearson"], ties.method = "average")
  rC  <- rank( met[, "cosine"],  ties.method = "average")
  avg <- (rP + rC) / 2
  names(avg) <- rownames(met)
  
  # update each pair (smaller avg rank wins)
  for (i in seq_along(pair_list)) {
    a <- avg[pair_list[[i]][1]]
    b2 <- avg[pair_list[[i]][2]]
    if (is.na(a) || is.na(b2)) next
    if (a < b2)       wins[i] <- wins[i] + 1L
    else if (a == b2) ties[i] <- ties[i] + 1L
  }
}

# summarize pairs
res <- tibble::tibble(
  comparison    = names(wins),
  wins          = as.integer(wins),
  ties          = as.integer(ties),
  N             = N,
  p_win         = (wins + 0.5 * ties) / N,
  BF_odds       = (p_win + 1e-8) / (1 - p_win + 1e-8),
  rows_per_draw = B,
  sample_frac   = sample_frac
) %>%
  dplyr::mutate(
    BF_band   = vapply(BF_odds, bf_band, character(1)),
    evidence  = vapply(BF_odds, bf_strength, character(1)),
    favored   = vapply(BF_odds, bf_favored, character(1)),
    tie_by_BF = BF_odds >= (1/3) & BF_odds <= 3
  )

cat(sprintf("\nBootstraps: %d | rows per draw: %d (sample_frac=%.2f)\n",
            N, B, sample_frac))
knitr::kable(
  res %>% dplyr::select(comparison, p_win, BF_odds, BF_band, evidence, favored, rows_per_draw, sample_frac, tie_by_BF),
  digits = 3,
  caption = sprintf("All pairwise comparisons among top %d teams (BF tie rule: 1/3 ≤ BF ≤ 3).", length(teams4))
)

# ---- 4-way tie verdict (only if we actually have 4 teams) ----
if (length(teams4) == 4) {
  # all six pair names among teams4
  all_pairs_4 <- combn(teams4, 2, FUN = function(p) paste0(p[1], " vs ", p[2]))
  res4 <- res %>% dplyr::filter(.data$comparison %in% all_pairs_4)
  four_way_tie <- nrow(res4) == 6 && all(res4$tie_by_BF, na.rm = TRUE)
  
  cat("\nFour-way tie check among: ",
      paste(teams4, collapse = " | "), "\n", sep = "")
  if (four_way_tie) {
    cat("Result: YES — all six pairwise Bayes factors are within [1/3, 3].\n")
  } else {
    cat("Result: NO — at least one pair has BF outside [1/3, 3].\n")
    # (optional) show which pairs broke the tie
    offenders <- res4 %>% dplyr::filter(!tie_by_BF)
    if (nrow(offenders) > 0) {
      knitr::kable(offenders %>% dplyr::select(comparison, BF_odds, BF_band, evidence),
                   digits = 3, caption = "Pairs preventing a 4-way tie")
    }
  }
}

# Print interpretation reference table once
print_bf_reference_table()


