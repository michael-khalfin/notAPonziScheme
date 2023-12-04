getwd()
setwd("/home/matthewbanschbach/R/CMOR462_Project2/")

rm(list=ls())

# install.packages("tidyverse") Already Installed

library(readr)
library(ggplot2)
library(tibble)
library(dplyr)
library(glue)


get_periods_data = function(){
  setwd("/home/matthewbanschbach/R/CMOR462_Project2/Parameter_Inputs/")
  ldf = list()
  listcsv = dir(pattern = "*.csv")
  
  for (k in 1:length(listcsv)){
    ldf[[k]] <- read.csv(listcsv[k])}
  
  ldf
}


get_num_periods = function(){
  setwd("/home/matthewbanschbach/R/CMOR462_Project2/Parameter_Inputs/")
  length(dir(pattern = "*.csv"))
}


process_mkt_returns = function(path, period, start){
  setwd("/home/matthewbanschbach/R/CMOR462_Project2/")
  mkt_ret = read_csv(path)
  mkt_ret = mkt_ret[start:period, ]  # Get the number
  # Here we grab (and make numeric) just the total market returns
  mkt_ret = mkt_ret$`Total Market`
  
  mkt_ret
}


build_covariance_matrix = function(betas, sigma_m_2, num_securities, resid_errors){
  # Initialize the diagonal matrix
  D_matrix = matrix(0, nrow=num_securities, ncol=num_securities)
  
  # Add the values to the diagonal matrix
  for(i in 1:num_securities){
    D_matrix[i,i] = resid_errors[i] ^ 2
  }
  
  
  
  covariance_matrix = matrix(betas) %*% t(matrix(betas))
  covariance_matrix = sigma_m_2 * covariance_matrix
  covariance_matrix = covariance_matrix + D_matrix
  
  covariance_matrix
}


generate_expected_i = function(betas, alphas, mean, period, start){
  setwd("/home/matthewbanschbach/R/CMOR462_Project2/Parameter_Outputs/")
  expected_returns = (betas*mean) + alphas
  e_path = glue("expected_return_{period}_{start}.csv")
  write.table(expected_returns, file=e_path, row.names = FALSE, col.names = FALSE)
}


output_covariance_matrix = function(covariance_matrix, period, start){
  setwd("/home/matthewbanschbach/R/CMOR462_Project2/Parameter_Outputs/")
  c_path = glue("covariance_matrix_{period}_{start}.csv")
  write.table(covariance_matrix, file=c_path, row.names = FALSE, col.names = FALSE, sep=",")
  
}


run_periods = function(start){
  
  period_data = get_periods_data()
  num_periods = get_num_periods()
  
  
  for(i in 1:num_periods){
    sec_ret = period_data[[i]]
    period = 59 + i
    mkt_ret = process_mkt_returns("market_return.csv", period, start)
    
    
    alphas = c()
    betas = c()
    resid_errors = c()
    
    num_securities = dim(sec_ret)[1]
    
    # Note that we run a regression for each security, and thus produce an estimate for each relevant parameter for each security,
    # before adding that parameter to a vector. That is to say, component i of the vector alphas, for example, refers to security i, not period i. 
    
    for(k in 1:num_securities){
      sec_i_returns = c(sec_ret[k, start:period+1])
      sec_i_returns = unlist(sec_i_returns)
      sec_i_reg_data = tibble(market=mkt_ret, security=sec_i_returns)
      sec_i_regression = lm(security ~ market, data=sec_i_reg_data)  # Runs the regression
      
      reg_sum_i = summary(sec_i_regression)  # Gets regression estimates
      a_i = reg_sum_i$coefficients[1,1]
      b_i = reg_sum_i$coefficients[1,1]
      sigma_i = reg_sum_i$sigma
      
      alphas = append(alphas, a_i)
      betas = append(betas, b_i)
      resid_errors = append(resid_errors, sigma_i)
    }
    
    # Get the market returns variance estimate
    sigma_m_2 = var(mkt_ret)
    E_r_m = mean(mkt_ret)
    
    generate_expected_i(betas, alphas, E_r_m, period, start)
    
    covariance_matrix = build_covariance_matrix(betas, sigma_m_2, num_securities, resid_errors)
    output_covariance_matrix(covariance_matrix, period, start)
    setwd("/home/matthewbanschbach/R/CMOR462_Project2/")
  }
}


run_periods(48)

