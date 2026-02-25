rm(list=ls())
library(tidyverse)

# Toy data: 
isToy = TRUE
Toy = ifelse(isToy, "toy_","")

# Get predictions: 
preds <- lapply(c("bergen", "oslo"), function(.city){
  readRDS(paste0("predictions/",ifelse(isToy, "toy_",""), .city,".rds")) %>% 
    mutate(area = .city)
}) %>% bind_rows()

# Brier bootstrap on test set: 
compute_brier_and_prauc <- function(data) {
  data %>%
    summarise(brier = 100*mean((pred - truth)^2),
              pr_auc = 100* (as.numeric(
                MLmetrics::PRAUC(y_pred=as.numeric(pred), y_true = truth))))
}


# Number of bootstrap samples
B <- 1000  

# Bootstrap and compute
bootstrap_results <- preds %>%
   pivot_longer(cols = Saturated:observed_forecast_gam,
                names_to = "model", 
                values_to = "pred") %>% 
  group_by(model, area) %>%
  summarise(bootstrap = list(map_dfr(1:B, 
                                     ~ slice_sample(pick(everything()), 
                                                    replace = TRUE,
                                                    n = n()) %>%
                                       compute_brier_and_prauc())),
            .groups = "drop") %>%
  unnest(bootstrap)

bootstrap_summary <- bootstrap_results %>%
  group_by(model, area) %>%
  summarise(brier_mean = mean(brier),
            brier_ci_lower = quantile(brier, 0.025),
            brier_ci_upper = quantile(brier, 0.975),
            prauc_mean = mean(pr_auc),
            prauc_ci_lower = quantile(pr_auc, 0.025),
            prauc_ci_upper = quantile(pr_auc, 0.975))


bootstrap_summary$model %>% unique()
bootstrap_summary$model[bootstrap_summary$model=="observed_forecast"] <- "observed-forecast"
bootstrap_summary$model[bootstrap_summary$model=="xgboost"] <- "xgboost"
bootstrap_summary$model[bootstrap_summary$model=="observed_forecast_gam"] <- "observed-forecast-gam"
bootstrap_summary$model[bootstrap_summary$model=="monkey"] <- "unconditional"
bootstrap_summary$model[bootstrap_summary$model == "GAM"] = "seasonal"


# PRAUC with constant probability: 
n = sum(preds$area=="oslo")
pr_auc_boot_oslo <- matrix(sample(as.numeric(preds[preds$area=="oslo", ]$truth), replace = TRUE, size = B*n), ncol = B)
pr_auc_oslo <- colMeans(pr_auc_boot_oslo)
pr_auc_oslo_metrics <- 100*as.numeric(c(mean(pr_auc_oslo), quantile(pr_auc_oslo, prob= c(0.025,0.975))))

n = sum(preds$area=="bergen")
pr_auc_boot_bergen <- matrix(sample(as.numeric(preds[preds$area=="bergen", ]$truth), replace = TRUE, size = B*n), ncol = B)
pr_auc_bergen <- colMeans(pr_auc_boot_bergen)
pr_auc_bergen_metrics <- 100*as.numeric(c(mean(pr_auc_bergen), quantile(pr_auc_bergen, prob= c(0.025,0.975))))


bootstrap_summary[bootstrap_summary$model == "unconditional" & bootstrap_summary$area =="oslo",c("prauc_mean","prauc_ci_lower", "prauc_ci_upper")] <- 
  tibble("prauc_mean"= pr_auc_oslo_metrics[1],
 "prauc_ci_lower" = pr_auc_oslo_metrics[2],
 "prauc_ci_upper"=pr_auc_oslo_metrics[3])
bootstrap_summary[bootstrap_summary$model == "unconditional" & bootstrap_summary$area =="bergen",c("prauc_mean","prauc_ci_lower", "prauc_ci_upper")] <- 
  tibble("prauc_mean"= pr_auc_bergen_metrics[1],
         "prauc_ci_lower" = pr_auc_bergen_metrics[2],
         "prauc_ci_upper"=pr_auc_bergen_metrics[3])
bootstrap_summary2<-bootstrap_summary %>% 
  filter(model %in% c(
    "Observed",
    "seasonal",
    "unconditional",
    "observed-forecast",
    "observed-forecast-gam", 
    "Saturated",
    "Stepwise",
    "Lasso",
    "xgboost",
    "CNN")) %>%
  mutate(model = factor(tolower(model), levels =tolower(rev(c(
    "Observed",
    "seasonal",
    "unconditional",
    "observed-forecast",
    "observed-forecast-gam", 
    "Saturated",
    "Stepwise",
    "Lasso",
    "xgboost",
    "CNN") )) ))

prauc_plot <- bootstrap_summary2 %>% 
  ggplot(aes(y=model,
             x=prauc_mean ,
             xmax=prauc_ci_lower,
             xmin=prauc_ci_upper,
             linetype = str_to_title(area))) +
  geom_errorbar(position = position_dodge(width=.5)) +
  geom_point(position = position_dodge(width=.5)) +
  xlab(expression("PRAUC ("*10^{-2}*")"))+
  theme_minimal()+
  theme(legend.title = element_blank(),
        panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        axis.title.y = element_blank(),
        legend.position = "top")
ggsave(prauc_plot,
       file = paste0("figures/fig_",Toy,"PRAUC_with_bootstrap_ci_minmax.pdf"),
       width = 8, height = 6)

