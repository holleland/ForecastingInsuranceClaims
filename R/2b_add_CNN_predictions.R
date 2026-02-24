# add the CNN predictions from Python: 
for(.city in c("bergen", "oslo")){
    pred <- readRDS(paste0("predictions/",ifelse(.isToy, "toy_",""), .city,".rds"))
    CNN = read_csv2(file = paste0("predictions/",ifelse(.isToy, "toy_",""),"CNN_", .city, "_test.csv"))$Probability %>% as.numeric()
    pred <- pred %>% dplyr::select(-CNN) %>% mutate(CNN = CNN,
                                                    unconditional = unconditional) %>%
      relocate(truth, Saturated, Stepwise,unconditional, Lasso, Median, Observed, `Observed-median`, GAM, CNN)
    saveRDS(pred, file = paste0("predictions/",ifelse(.isToy, "toy_",""), .city,".rds"))
}
