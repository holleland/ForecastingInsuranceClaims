rm(list=ls())
library(tidyverse)
library(tidymodels)
library(MASS)
library(pROC)
library(mgcv)
library(xgboost)

theme_set(theme_bw()+
            theme(panel.grid.minor =element_blank(),
                  strip.background = element_rect(fill = "transparent", 
                                                  color = "transparent"))
)

# isToy = TRUE 
rawdata <- readRDS("data/toydata.rds")


for(.city in c("bergen","oslo")){
  print(paste(ifelse(isToy, "Toy", "data"),"-",
        .city))
data <- rawdata %>% 
    mutate(yday = lubridate::yday(date)/365) %>% 
    filter(area==.city)%>%
    mutate(set = ifelse(date >= as.Date("2019-12-30") | 
                          (area == "bergen" & date %in% 
                          seq(as.Date("2017-12-11"),
                              as.Date("2017-12-31"), 
                              "1 day")),"test","train")
                        ,
           claim_cat = factor(claim_cat, levels = c("few claims","many claims")))
set.seed(1234)
train <- data %>% filter(set == "train")
test <- data %>% filter(set == "test")

table(test$claim_cat)/nrow(test)
table(train$claim_cat)/nrow(train)

readr::write_csv2(train %>% mutate(lead_time = as.numeric(lead_time)) %>% 
                     dplyr::select(-contains("s(yday)")) %>% ungroup(), 
                   file = paste0("data/", ifelse(isToy, "toy_",""), "train_",.city,".csv"), quote = "none")
readr::write_csv2(test %>% mutate(lead_time = as.numeric(lead_time)) %>% 
                     dplyr::select(-contains("s(yday)")) %>% ungroup(), 
                   file = paste0("data/",ifelse(isToy, "toy_",""),"test_", .city,".csv"),
                   quote = "none")

 # Logistic regression models: 
 obs.model <- glm(claim_cat~obs, data = train ,
                  family = binomial(link = "logit"))
 
 median.model <- glm(claim_cat~`26`, data = train %>%ungroup() %>%  dplyr::select(-date, -lead_time,  -area, -obs) ,
                     family = binomial(link = "logit"))
 
 full.model <- glm(claim_cat~., data = train %>%ungroup() %>%  dplyr::select(-date, -lead_time,  -area, -obs, -yday,-set),
                   family = binomial(link = "logit"))
 step.model <- stepAIC(full.model, direction = "both", 
                       trace = TRUE, 
                       k = log(nrow(train)))
 # --- observed-forecast models: ----
 obs.gam <- mgcv::gam(claim_cat ~ s(obs), data = train,
                      family = binomial(link = "logit"))
 test_long <- test %>%
   rename("observed"="obs") %>% 
   pivot_longer(cols = 5:55, names_to = "ensemble", values_to = "obs") 
 test_long$pred <- predict(obs.model, newdata= test_long, 
                           type = "response")
 test_long$pred.gam <- predict(obs.gam, newdata= test_long, 
                               type = "response")
 
 observed_forecast <- test_long %>% group_by(date) %>% summarize(pred = mean(pred))%>% pull(pred)
 observed_forecast_gam <- test_long %>% group_by(date) %>% summarize(pred.gam = mean(pred.gam)) %>% pull(pred.gam)

 train_long <- train %>% 
   rename("observed"="obs") %>% 
   pivot_longer(cols = 5:55, names_to = "ensemble", values_to = "obs")
 train_long$pred <- predict(obs.model, newdata= train_long, 
                            type = "response")
 train_long$pred.gam <- predict(obs.gam, newdata= train_long, 
                                type = "response")
 observed_forecast_fitted <- train_long %>% group_by(date) %>% summarize(pred = mean(pred))%>% pull(pred)
 observed_forecast_gam_fitted <- train_long %>% group_by(date) %>% summarize(pred.gam = mean(pred.gam)) %>% pull(pred.gam)
 
# seasonal model: 
gammod <- gam(formula = claim_cat~s(yday, bs = "cp"), 
              data = train,
              family=binomial(link = "logit"))
gam.pred <- predict(gammod, newdata = test, type ="response")
gam.fitted <- predict(gammod, newdata=train, type = "response")

# lasso and xgboost using tidymodels
# Recipe: 
claim_rec <- recipe(claim_cat ~ ., data = train)  %>% 
  update_role(date, new_role = "ID") %>% 
  update_role(lead_time, new_role = "ID") %>% 
  update_role(area, new_role = "ID") %>% 
  update_role(yday, new_role = "ID")%>% 
  update_role(obs, new_role = "ID")%>% 
  update_role(set, new_role = "ID")%>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors())
claim_rec_minmax <- recipe(claim_cat ~ `1`+ `26`+ `51`, data = train) 

# Spec
lasso_spec <- logistic_reg(penalty = 0.1, mixture = 1) %>% 
  set_engine("glmnet")
xgb_spec <- boost_tree(
  trees = 1000,  # Number of trees
  tree_depth = tune(),  # Depth of each tree
  learn_rate = tune(),  # Learning rate
  loss_reduction = tune(),  # Minimum loss reduction
  sample_size = tune(),  # Row sampling
  mtry = tune(),  # Feature sampling
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification") 

# Workflow: 
wf <- workflow() %>% 
  add_recipe(claim_rec)
xgb_wf <- workflow() %>% 
  add_recipe(claim_rec_minmax) %>% 
  add_model(xgb_spec)

xgb_grid <- grid_space_filling(
  tree_depth(range = c(3, 10)),  
  learn_rate(range = c(0.01, 0.3)),  
  loss_reduction(range = c(0.0001, 1)),  
  sample_prop(range = c(0.5, 1)),
  finalize(mtry(), train),  # mtry set based on data size
  size = 100  # Number of grid points
)

lasso_fit <- wf %>% 
  add_model(lasso_spec) %>% 
  fit(data = train)

# --- Tuning --- 
set.seed(1234)
claim_boot <- bootstraps(train, strata = claim_cat)

tune_spec <- logistic_reg(penalty = tune(), mixture = 1) %>% 
  set_engine("glmnet")

lambda_grid <- grid_regular(penalty(), 
             levels = 50)

set.seed(2020)
lasso_grid <- tune_grid(
  wf %>%  add_model(tune_spec),
  resamples = claim_boot,
  grid = lambda_grid
)

lasso_grid %>% 
  collect_metrics() %>% 
  ggplot(aes(x=penalty, y = mean, color = .metric))+
  geom_errorbar(aes(ymin = mean-std_err,ymax = mean+std_err), 
                alpha = 0.5)+
  geom_line()+
  facet_wrap( ~.metric, scales = "free", nrow=3)+
  scale_x_log10()+
  theme(legend.position = "none")

highest_auc <- lasso_grid %>% 
  select_best(metric = "roc_auc")
final_lasso <- finalize_workflow(wf %>% add_model(tune_spec),
                                 highest_auc) 

lasso.fitted <- final_lasso %>%  
  fit(train) %>% 
  predict(new_data = train, type = "prob") %>% pull(`.pred_many claims`)
lasso_preds <- final_lasso %>%  
  fit(train) %>% 
  predict(new_data = test, type = "prob") %>% pull(`.pred_many claims`)

# XGBoost tuning: 
set.seed(123)
cv_folds <- vfold_cv(train, v = 5, strata = claim_cat)  # 5-fold CV
t0<-Sys.time()
tuned_results <- tune_grid(
  xgb_wf,
  resamples = cv_folds,
  grid = xgb_grid,
  metrics = metric_set(roc_auc, accuracy)
)
print(Sys.time()-t0)

# View the best hyperparameters
best_params <- select_best(tuned_results, metric = "roc_auc")
best_params


final_xgb <- finalize_workflow(xgb_wf, best_params)

# Results
final_fit_xgb <- final_xgb %>% fit(data = train)
xgb_fitted <- final_fit_xgb %>% 
  predict(new_data = train, type = "prob") %>% 
  pull(`.pred_many claims`)
xgb_pred<- final_fit_xgb %>% 
  predict(new_data = test, type = "prob")  %>% 
  pull(`.pred_many claims`)


train_predictions <- tibble(
  "truth" = train$claim_cat == "many claims",
  "Saturated" = predict(full.model, newdata=train, type ="response"),
  "Stepwise"   =  predict(step.model, newdata = train, type = "response"),
  "unconditional" = mean(train$claim_cat=="many claims"),
  "Lasso" = lasso.fitted,
  "Median" = predict(median.model, newdata=train, type = "response"),
  "Observed" = predict(obs.model, newdata=train, type = "response"),
  "GAM" = gam.fitted,
  "xgboost" = xgb_fitted,
  "observed_forecast" = observed_forecast_fitted,
  "observed_forecast_gam" = observed_forecast_gam_fitted
  
)

test_predictions <- tibble(
  "truth" = test$claim_cat == "many claims",
  "Saturated" = predict(full.model, newdata=test, type ="response"),
  "Stepwise"   =  predict(step.model, newdata = test, type = "response"),
  "unconditional" = mean(train$claim_cat=="many claims"),
  "Lasso" = lasso_preds,
  "Median" = predict(median.model, newdata=test, type = "response"),
  "Observed" = predict(obs.model, newdata=test, type = "response"),
  "Observed-median" = predict(obs.model, newdata=test %>% mutate(obs =`26`), type = "response"),
  "GAM" = gam.pred,
  "xgboost" = xgb_pred,
  "observed_forecast" = observed_forecast,
  "observed_forecast_gam" = observed_forecast_gam
)


saveRDS(test_predictions, 
                  file = paste0("predictions/",ifelse(isToy, "toy_",""), .city,".rds"))
saveRDS(train_predictions, 
        file = paste0("predictions/",ifelse(isToy, "toy_",""), .city,"_train.rds"))

}

print("------ COMPLETED. ----------")






