# ===========================
# Model 2 (no predictors; ordination)
# GLLVM with NB counts, 2 latent variables, no X
# ===========================

# ---- Paths ----
DATA_CSV <- "C:/Users/olatu/OneDrive/Desktop/Msc Project/Dataset.csv"
OUT_DIR  <- "C:/Users/olatu/OneDrive/Desktop/Msc Project/gllvm_out_model2"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---- Packages ----
suppressPackageStartupMessages({
  library(gllvm); library(readr); library(dplyr); library(tidyr); library(ggplot2)
})

# ---- Load & tidy ----
df <- readr::read_csv(DATA_CSV, show_col_types = FALSE)

# Clean header names (handles stray spaces / NBSP)
nm <- names(df); nm <- gsub("\u00A0", " ", nm, useBytes = TRUE); nm <- trimws(nm)
names(df) <- nm

needed <- c("Plot Id","Tree Species","Forest Id","Dbh(cm)","Ht(m)","Density")
miss <- setdiff(needed, names(df))
if (length(miss)) stop("Missing columns: ", paste(miss, collapse=", "))

df <- df |>
  mutate(`Plot Id` = trimws(as.character(`Plot Id`)),
         `Tree Species` = trimws(as.character(`Tree Species`)))

# ---- Response matrix Y (plots × species counts) ----
Y <- df |>
  count(`Plot Id`, `Tree Species`, name = "abund") |>
  pivot_wider(
    id_cols     = `Plot Id`,
    names_from  = `Tree Species`,
    values_from = abund,
    values_fill = 0
  ) |>
  arrange(`Plot Id`)

plot_ids <- Y[["Plot Id"]]
Y_mat    <- as.matrix(Y[, -which(names(Y) == "Plot Id")])
rownames(Y_mat) <- plot_ids
write.csv(Y, file.path(OUT_DIR, "Y_species_by_plot.csv"), row.names = FALSE)
cat("Y matrix:", nrow(Y_mat), "plots ×", ncol(Y_mat), "species\n")

# ===========================
# Fit Model 2 (no predictors; ordination)
# ===========================
set.seed(123)
fit_m2 <- gllvm(
  y      = Y_mat,
  family = "negative.binomial",
  num_lv = 2,       
  method = "VA"      
)

# ---- Outputs ----
# Summary & diagnostics
sink(file.path(OUT_DIR, "model2_summary.txt")); print(summary(fit_m2)); sink()

png(file.path(OUT_DIR, "model2_diagnostics.png"), width=1800, height=1200, res=180)
plot(fit_m2)   # Dunn–Smyth residuals & QQ
dev.off()

# Ordinations (sites only, and biplot with top species)
png(file.path(OUT_DIR, "model2_ordination_sites.png"), width=1400, height=1000, res=150)
ordiplot(fit_m2, biplot = FALSE)
dev.off()

png(file.path(OUT_DIR, "model2_ordination_biplot_15.png"), width=1400, height=1000, res=150)
ordiplot(fit_m2, biplot = TRUE, ind.spp = 15) 
dev.off()

# Site scores (LV coordinates for each plot)
m2_sites <- as.data.frame(getLV(fit_m2))
colnames(m2_sites) <- paste0("LV", seq_len(ncol(m2_sites)))
m2_sites$PlotId <- rownames(Y_mat)
write.csv(m2_sites, file.path(OUT_DIR, "model2_site_scores.csv"), row.names = FALSE)

# Species loadings 
m2_load <- as.data.frame(getLoadings(fit_m2))
colnames(m2_load) <- paste0("LV", seq_len(ncol(m2_load)))
m2_load$Species <- rownames(m2_load)
write.csv(m2_load, file.path(OUT_DIR, "model2_species_loadings.csv"), row.names = FALSE)


