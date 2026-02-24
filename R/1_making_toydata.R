# Making toy data:
library(tidyverse)
# observed model coefficients for Oslo:
theta_oslo   <- c(-4.6484184,  0.1349749)
theta_bergen <- c(-5.43247741,  0.09294646)
set.seed(1234)
forecasts <- readRDS(file = "data/forecast_data.rds")
forecasts <- forecasts %>%  
  mutate(eta = ifelse(area == "bergen", theta_bergen[1]+theta_bergen[2]*obs,
                      theta_oslo[1]+theta_oslo[2]*obs)) %>% 
  mutate(prob = exp(eta)/(1+exp(eta)))



claims = stats::rbinom(n= nrow(forecasts),
                       prob = as.numeric(forecasts$prob), 
                       size =1)
forecasts$claim_cat = ifelse(claims==1, 
                             "many claims", 
                             "few claims")
forecasts <- forecasts %>%  
  dplyr::select(-eta, -prob)
table(forecasts$claim_cat)/nrow(forecasts)
saveRDS(forecasts, file = "data/toydata.rds")

