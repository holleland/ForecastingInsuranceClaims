# ---------- RELIABILITY ----------
rm(list=ls())
library(tidyverse)
theme_set(theme_minimal())
# Toy data: 
isToy = FALSE
Toy = ifelse(isToy, "toy_","")

# Get predictions: 
pred <- lapply(c("bergen", "oslo"), function(.city){
  readRDS(paste0("predictions/",ifelse(isToy, "toy_",""), .city,".rds")) %>% 
    mutate(area = .city)
}) %>% bind_rows()



reliabilityboth <- pred %>% 
  rename("Seasonal"="GAM",
         #"One Forecast" = "oneforecast",
         "Observed-forecast" ="observed_forecast" ,
         "Observed-forecast-gam" ="observed_forecast_gam",
         "XGBoost" ="xgboost") %>% 
  pivot_longer(Saturated:`Observed-forecast-gam`) %>% 
  filter(!str_detect(name, "unconditional")) %>% 
  group_by(name,
           area) %>% 
  mutate(#predbin = (floor(value*10)+1/(2*10))/10,
    predbin = as.character(cut_interval(log(value), n=10))) %>% 
  group_by(name,
           predbin,
           area) %>% 
  summarize(.obs = mean(truth), n=n()) %>% 
  mutate(offset = mean(pred$truth)) 

# Extract middle points: 
reliabilityboth$predbin <- unlist(
  lapply(strsplit(substr(reliabilityboth$predbin, 2, 
                         nchar(reliabilityboth$predbin)-1), ","),
         function(x)mean(as.numeric(x)))
)

mod <- glm(log(.obs) ~ 0 + area:name:predbin, 
           data = reliabilityboth %>% filter(.obs >0), 
           weights = n
)

pred.tbl <- expand.grid(name = unique(reliabilityboth$name),
                        area = c("bergen", "oslo"),
                        predbin= seq(-25.1,-0.2,0.01)) %>% as_tibble() 
pred.tbl$pred <- predict(mod, newdata= pred.tbl,type = "response")
ss <- summary(mod)$coef
plist <- list()
reliabilityboth<-reliabilityboth %>% 
filter(name %in% c(
  "Observed",
  "Seasonal",
  #"Unconditional",
  #"One Forecast",
  #"Median","Observed-median",
  "Observed-forecast",
  "Observed-forecast-gam", 
  "Saturated",
  "Stepwise",
  "Lasso",
  "XGBoost",
  "CNN")) %>%
  mutate(name = factor(tolower(name), levels =tolower(c(
    "Observed",
    "Seasonal",
    #"Unconditional",
    "Median","Observed-median",
    "Observed-forecast",
    "Observed-forecast-gam", 
    "Saturated",
    "Stepwise",
    "Lasso",
    "XGBoost",
    "CNN") )))

reliabilityboth %>% 
  filter(.obs>0) %>% 
  group_by(area, name) %>% 
  summarize(RSS=mean((log(.obs)-predbin)^2)) %>% 
  pivot_wider(names_from = area, values_from = RSS)

for(.area in c("bergen", "oslo")){
  p <- pred %>% filter(area==.area) %>% pull(truth) %>% mean()
  plist[[.area]] <-reliabilityboth %>% 
    filter(area == .area, .obs >0) %>% 
    ggplot(aes(x=predbin, y = log(.obs))) + 
    geom_point(aes(size = n)) +
    geom_abline(linetype = "dashed", linewidth =.9)+
    facet_wrap(~name, nrow = 3, scales = "free") +
    #geom_line(data=pred.tbl %>% filter(area == .area), aes(y = pred), col = "blue", lwd = 1.2)+
    scale_x_continuous("log(forecasted probability)")+#, 
                       #limits = c(0,1),
                       #expand =c(0,0),
                       #breaks = seq(0,1,0.25),
                       #labels = c("0",".25",".5",".75","1"))+
    scale_y_continuous("log(observed relative frequency)")+#, 
                       #limits = c(0,1),
                       #expand =c(0,0),
                       #breaks = seq(0,1,0.25),
                       #labels = c("0",".25",".5",".75","1"))+
    
    geom_ribbon(data=filter(pred.tbl, 
                            predbin >=p,
                            area == .area,
                            name == "Observed") %>% rename(tull = name),y = NA,
                aes(ymin = pred, ymax = 1),
                fill="grey30", alpha=0.05) +
    # geom_ribbon(data=filter(pred.tbl, 
    #                         predbin <p,
    #                         area == .area,
    #                         name == "Observed") %>%
    #               rename(tull = name),y = NA,
    #             aes(ymax = pred, ymin = 0),
    #             fill="blue", alpha=0.05) +
    geom_vline(xintercept = log(mean(pred$truth)), lty = 2)+
    geom_hline(yintercept = log(mean(pred$truth)), lty = 2) +
    ggtitle(str_to_title(.area))+
    geom_smooth(
      method = "lm",
      mapping = aes(weight = n), 
      se = TRUE,
      alpha = 0,
      color = "blue",
      fill = "grey70",
      show.legend = FALSE
    )+
    theme(strip.text = element_text(family = "mono", face = "plain", size = 12),
          legend.position = "none")
  }
ggpubr::ggarrange(plist$bergen +
                    theme(axis.title.x = element_blank()), 
                  plist$oslo, nrow = 2,
                  heights = c(.95,1))
ggsave(
  paste0("figures/",Toy, "reliability curve_both_log_minmax.pdf"), width = 10, height = 14)
