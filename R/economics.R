library(tidyverse)
library(cowplot)


metric_simulator <- function(sd_seed, samples = 50){
  performance <- tibble(truth = rbinom(samples, size = 1, prob = .5), 
                        prediction = rnorm(samples, mean = truth, sd = sd_seed)) %>% 
    mutate(prediction = if_else(prediction > 1, 1, prediction), 
           prediction = if_else(prediction < 0, 0, prediction),
           truth =  truth %>% factor())
  return(performance)
}

calculate_saving <- function(df){
  
metric_scored <- tibble(n_screen_0 = (df$n_target/df$p),
                          C0 = n_screen_0*df$cost_0,
                          n_screen_1 = (df$n_target/(df$p*df$sens)), # patients that need to be screened
                          C1 = n_screen_1*df$p*df$sens*df$cost_0 + # cost for true positives
                          n_screen_1*(1-df$p)*(1-df$spec)*df$cost_0 + # cost for false positives
                          n_screen_1*df$cost_1, # cost of screening
                          ppv = (df$p*df$sens)/(df$p*df$sens+(1-df$spec)*(1-df$p)),
                          r_1 = n_screen_1*df$p*(1-df$sens),
                          dC = C0-C1) #%>% 
  #mutate(dC = if_else(dC <0, 0, dC))
                          
  return(metric_scored)
}

#############
# legacy 
# define table of possible model performance metrics 
metric_table <- tibble(sens = seq(0,1,0.01), 
                       spec = seq(0,1,0.01)) %>%
  complete(sens, spec)

simulate_saving <- function(n_target = 300, 
                            p = 0.03, 
                            cost_0 = 250, 
                            cost_1 = 5, 
                            sens = 0.5, 
                            spec = 0.5, 
                            metric = metric_table){
  
  
  
  metric_scored <- metric %>% 
    mutate(n_screen_0 = (n_target/p),
           C0 = n_screen_0*cost_0,
           n_screen_1 = (n_target/(p*sens)), # patients that need to be screened
           C1 = n_screen_1*p*sens*cost_0 + # cost for true positives
             n_screen_1*(1-p)*(1-spec)*cost_0 + # cost for false positives
             n_screen_1*cost_1, # cost of screening
           ppv = (p*sens)/(p*sens+(1-spec)*(1-p)),
           r_1 = n_screen_1*p*(1-sens),
           dC = C0-C1, 
           dC = if_else(dC <0, 0, dC)
    )
  return(metric_scored)
}
