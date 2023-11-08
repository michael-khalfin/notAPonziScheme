getwd()
setwd("/home/matthewbanschbach/R/CMOR462_Project2/")

rm(list=ls())

install.packages("tidyverse")

library(readr)
library(ggplot2)
library(tibble)
library(dplyr)

# Read in the cleaned market returns csv file
mkt_ret = read_csv("market_return.csv")

# Get soley the first 60 periods-- note that we removed all entries before 
# 2017, so here we just need to make sure we dont get any 2022, 2023 entries
mkt_ret = mkt_ret[1:60,]


# Now, read in the data on the securities
sec_ret = read_csv("market_returns.csv")  # We are conscerned primarily with these
sec_cap = read_csv("capitalizations.csv")

# Get the number of securities in the csv- good to do it this way in case we adjust the process of producing it
num_securities = dim(sec_ret)[1]


# Here we grab (and make numeric) just the total market returns
mkt_returns = c(matrix(mkt_ret$`Total Market`))


# TESTING PURPOSES ONLY --- WORKS ON THE FIRST SECURITY

# sec_1_returns = unlist(matrix(sec_ret[1, 2:61]))
# sec_1_reg_data = tibble(market=mkt_returns, security=sec_1_returns)
# sec_1_regression = lm(security ~ market, data=sec_1_reg_data)
# reg_sum = summary(sec_1_regression)
# epsilon_1 = reg_sum$sigma
# a_1 = reg_sum$coefficients[1, 1]
# b_1 = reg_sum$coefficients[2,1]


# Data Structures for Storing Estimates
alphas = c()
betas = c()
resid_errors = c()

# Note that we run a regression for each security, and thus produce an estimate for each relevant parameter for each security,
# before adding that parameter to a vector. That is to say, component i of the vector alphas, for example, refers to security i, not period i. 

for(i in 1:num_securities){
  sec_i_returns = unlist(matrix(sec_ret[i, 2:61]))
  sec_i_reg_data = tibble(market=mkt_returns, security=sec_i_returns)
  sec_i_regression = lm(security ~ market, data=sec_i_reg_data)  # Runs the regression

  reg_sum_i = summary(sec_i_regression)  # Gets regression estimates
  a_i = reg_sum_i$coefficients[1,1]
  b_i = reg_sum_i$coefficients[1,1]
  sigma_i = reg_sum_i$sigma
  
  alphas = append(alphas, a_i)
  betas = append(betas, b_i)
  resid_errors = append(resid_errors, sigma_i)
}
reg_sum_i

# Clean up the workspace

rm(a_i, b_i, sigma_i)
rm(sec_i_reg_data, sec_i_regression, reg_sum_i, i)
rm(sec_i_returns)

# Initialize the diagonal matrix
D_matrix = matrix(0, nrow=num_securities, ncol=num_securities)

# Add the values to the diagonal matrix

for(i in 1:num_securities){
  D_matrix[i,i] = resid_errors[i] ^ 2
}

# Clean up workspace
rm(i)

# Get the market returns variance estimate
sigma_m_2 = var(mkt_returns)

# Build the covariance matrix
covariance_matrix = matrix(betas) %*% t(matrix(betas))
covariance_matrix = sigma_m_2 * covariance_matrix
covariance_matrix = covariance_matrix + D_matrix

# Build the expected returns vector
E_r_m = mean(mkt_returns)
expected_returns = (betas*E_r_m) + alphas

write.table(covariance_matrix, file="covariance_matrix.txt", row.names = FALSE, col.names = FALSE)
write.table(sec_ret$names, file="names.txt", row.names =  FALSE, col.names = FALSE)











