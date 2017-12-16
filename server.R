library(ggplot2)
library(dplyr)
library(devtools)
library(coda)
library(shiny)

###############################################################################
set.seed(2017)
data = c(70, 80, 75, 83, 72)
data = data.frame(data)
data = sort(data[,1])
tau=c(0.8,rep(0.4,3-1))
N=3
################################################################################
meandata <- mean(data)
standarddeviationdata <- sd(data)
mediandata <- median(data)
rangedata <- max(data) - min(data)

#######################################################################################################################
# The run_ABC_rejection function returns a data table with all simulation records
# Input: 
#       prior_params: prior distrinution parameters
#       n: the total number of simulations
#       threshold: acceptane threshold
# Output: 
#       a data table 'sim' that contains all simulation records

run_ABC_rejection = function(prior_mean, prior_sd, prior_s, prior_r, iters, threshold, metric, summary) {
  #sample n pairs of parameters from prior distribution
  tmp_mean <- rnorm(iters, mean = prior_mean, sd = prior_sd) #改成class
  tmp_precision <- rgamma(iters, prior_s, prior_r)
  
  sim_mean = rep(NA, iters)
  sim_precision = rep(NA, iters)
  sim_deviation = rep(NA, iters)
  sim_acceptance = rep(NA, iters)
  
  for (i in 1:iters){
    # For each lambda, generate a simulated data with 13 cases
    samples = rnorm(5, tmp_mean[i], sqrt(1/tmp_precision[i]))
    sorted_samples <- sort(samples)
    sim_mean[i] = mean(samples)
    sim_precision[i] = 1/(sd(samples))^2
    
    if ("data" %in% summary) {
      data_stat <- data
      sample_stat <- sorted_samples
    } else {
      data_stat <- c()
      sample_stat <- c()
      for (s in summary) {
        if (s == "mean") {
          data_stat <- c(data_stat, meandata)
          sample_stat <- c(sample_stat, mean(sorted_samples))
        } else if (s == "sd") {
          data_stat <- c(data_stat, standarddeviationdata)
          sample_stat <- c(sample_stat, sd(sorted_samples))
        } else if (s == "median") {
          data_stat <- c(data_stat, mediandata)
          sample_stat <- c(sample_stat, median(sorted_samples))
        } else if (s == "range") {
          data_stat <- c(data_stat, rangedata)
          sample_stat <- c(sample_stat, max(sorted_samples) - min(sorted_samples))
        }
      }
    }
    
    # compute the deviation by takeing the Euclidean distance between observed summary statistics
    
    if (metric == "L1Norm") {
      sim_deviation[i] = dist(rbind(data_stat, sample_stat), method = "manhattan")
    } else if (metric == "L2Norm") {
      sim_deviation[i] = dist(rbind(data_stat, sample_stat))
    }
    
    # Accept if deviation is smaller than threshhold; reject otherwise.
    if(sim_deviation[i] < threshold)
      sim_acceptance[i] = TRUE
    else
      sim_acceptance[i] = FALSE
  }
  sim = data.frame(index = c(1:iters), tmp_mean, tmp_precision, sim_mean, sim_precision, sim_deviation, sim_acceptance)
  return(sim)
}

#######################################################################################################################


# The run_ABC_SMC function returns a data table with all simulation records
# Input: 
#       prior_params: prior distrinution parameters
#       n: the total number of simulations
#       N: the total number of rounds
#       tau: acceptane percentage
# Output: 
#       a data table 'sim' that contains all simulation records

run_ABC_SMC = function(prior_mean, prior_sd, prior_s, prior_r, n,N,tau,metric,summary) {
  #sample n pairs of parameters from prior distribution
  tmp_mean <- rnorm(n, mean = prior_mean, sd = prior_sd)
  tmp_precision <- rgamma(n, prior_s, prior_r)
  dist_vals <- rep(NA,n)
  Q=n*tau[1]
  epsilon=rep(NA,N)
  
  #First round:draw N data sets and compare
  for(i in 1:n){
    curr_dat <- rnorm(length(data), tmp_mean[i], sqrt(1/tmp_precision[i]))
    curr_dat <- sort(curr_dat)
    if ("data" %in% summary) {
      data_stat <- data
      sample_stat <- curr_dat
    } else {
      data_stat <- c()
      sample_stat <- c()
    }
    
    for (s in summary){
      if (s == "mean") {
        data_stat <- c(data_stat, mean(data))
        sample_stat <- c(sample_stat, mean(curr_dat))
      } else if (s == "sd") {
        data_stat <- c(data_stat, sd(data))
        sample_stat <- c(sample_stat, sd(curr_dat))
      } else if (s == "median") {
        data_stat <- c(data_stat, median(data))
        sample_stat <- c(sample_stat, median(curr_dat))
      } else if (s == "range") {
        data_stat <- c(data_stat, max(data)-min(data))
        sample_stat <- c(sample_stat, max(curr_dat) - min(curr_dat))
      }
      
    }
    
    
    if (metric=="L1Norm"){
      dist_vals[i] = dist(rbind(data_stat, sample_stat), method = "manhattan")
    } else if (metric == "L2Norm") {
      dist_vals[i] = dist(rbind(data_stat, sample_stat))
    }
  }
  
  dist_indexes <- sort(dist_vals, index.return=T)
  save_indexes <- dist_indexes$ix[1:Q]
  epsilon[1] <- dist_indexes$x[Q] # first epsilon is the max dist value
  saved_means <- tmp_mean[save_indexes]
  saved_precision <- tmp_precision[save_indexes]
  
  #round_twoToN<-list(N-1)
  for (r in 2:N){
    curr_num_saved <- 0
    dist_vals <- rep(NA,Q)
    tmp_saved_means <- rep(NA,Q)
    tmp_saved_precision <- rep(NA,Q)
    
    while(curr_num_saved < Q){
      curr_mean <- sample(saved_means,1)
      curr_mean <- curr_mean + runif(1,-0.1,0.1)
      curr_precision <- sample(saved_precision,1)
      curr_precision <- curr_precision + runif(1,-0.05,0.05)
      
      if (curr_precision<0){
        curr_precision=0.01
      }
      
      curr_dat <- rnorm(length(data),curr_mean,sqrt(1/curr_precision))
      curr_dat <- sort(curr_dat)
      
      if ("data" %in% summary) {
        data_stat <- data
        sample_stat <- curr_dat
      } else {
        data_stat <- c()
        sample_stat <- c()
      }
      
      for (s in summary){
        if (s == "mean") {
          data_stat <- c(data_stat, mean(data))
          sample_stat <- c(sample_stat, mean(curr_dat))
        } else if (s == "sd") {
          data_stat <- c(data_stat, sd(data))
          sample_stat <- c(sample_stat, sd(curr_dat))
        } else if (s == "median") {
          data_stat <- c(data_stat, median(data))
          sample_stat <- c(sample_stat, median(curr_dat))
        } else if (s == "range") {
          data_stat <- c(data_stat, max(data)-min(data))
          sample_stat <- c(sample_stat, max(curr_dat) - min(curr_dat))
        }
        
      }
      
      
      if (metric=="L1Norm"){
        curr_dist = dist(rbind(data_stat, sample_stat), method = "manhattan")
      } else if (metric == "L2Norm") {
        curr_dist = dist(rbind(data_stat, sample_stat))
      
      }
      
      # curr_dist <- dist(rbind(dat,curr_dat))
      
      # Only save value if it is better than the previously saved max
      if(curr_dist < epsilon[r-1]){
        curr_num_saved <- curr_num_saved + 1
        dist_vals[curr_num_saved] <- curr_dist
        tmp_saved_means[curr_num_saved] <- curr_mean
        tmp_saved_precision[curr_num_saved] <- curr_precision
      }
    }
    
    dist_indexes <- sort(dist_vals, index.return=T)
    save_indexes <- dist_indexes$ix[1:(tau[r]*Q)]
    epsilon[r] <- dist_indexes$x[(tau[r]*Q)]
    saved_mean<-tmp_saved_means[save_indexes]
    saved_precision <-tmp_saved_precision[save_indexes]
    
  }
  acceptance=rep(TRUE,length(saved_mean))
  
  sim<-data.frame(tmp_mean=saved_mean,tmp_precision=saved_precision,sim_acceptance=acceptance)
  return(sim)
}

#######################################################################################################################

# The ABC-MCMC acceptance function

ABC_acceptance <- function(par,threshold,summary,metric){
  
  
  # prior to avoid negative standard deviation
  if (par[2] <= 0) {
    return(F) 
  }
  
  # stochastic model generates a sample for given par
  samples <- rnorm(5, mean =par[1], sd = sqrt(1/par[2]))
  sorted_samples <- sort(samples)
  
  if ("data" %in% summary) {
    data_stat <- data
    sample_stat <- sorted_samples
  } else {
    data_stat <- c()
    sample_stat <- c()
    for (s in summary) {
      if (s == "mean") { 
        data_stat <- c(data_stat, meandata)
        sample_stat <- c(sample_stat, mean(sorted_samples))
      } else if (s == "sd") {
        data_stat <- c(data_stat, standarddeviationdata)
        sample_stat <- c(sample_stat, sd(sorted_samples))
      } else if (s == "median") {
        data_stat <- c(data_stat, mediandata)
        sample_stat <- c(sample_stat, median(sorted_samples))
      } else if (s == "range") {
        data_stat <- c(data_stat, rangedata)
        sample_stat <- c(sample_stat, max(sorted_samples) - min(sorted_samples))
      }
    }
  }
  
  # comparison with the observed summary statistics
  samplemean <- mean(samples) 
  sampleprecision <- 1/(sd(samples)^2) 
  
  # compute the deviation by takeing the Euclidean distance between observed summary statistics
  
  if (metric == "L1Norm") {
    sim_deviation = dist(rbind(data_stat, sample_stat), method = "manhattan")
  } else if (metric == "L2Norm") {
    sim_deviation = dist(rbind(data_stat, sample_stat))
  }
  
  # Accept if deviation is smaller than threshhold; reject otherwise.
  if(sim_deviation < threshold){
    return(list(T,samplemean,sampleprecision)) 
  } else {
    return(list(F,samplemean,sampleprecision))
  }
}

#######################################################################################################################
run_MCMC_ABC <- function(prior_mean, prior_sd, prior_s, prior_r, iters, threshold, metric, summary){
  
  # parameter proposals 
  tmp_mean <- rep(NA, iters)
  tmp_precision <- rep(NA, iters)
  
  # generated datasets 
  sim_mean = rep(NA, iters)
  sim_precision = rep(NA, iters)
  sim_acceptance = rep(NA, iters)
  
  # initial values for parameters
  start_mu = rnorm(1,prior_mean,prior_sd)
  start_tau = rgamma(1,prior_s,prior_r)
  startvalue=c(start_mu,start_tau)
  
  chain = array(dim = c(iters+1,2))
  chain[1,] = startvalue
  
  for (i in 1:iters){
    
    # proposal function: draw from normal distribution 
    # proposal = rnorm(2, mean = chain[i,], sd= c(0.7,0.3))
    # uniform proposal function
    proposal = chain[i,]+runif(2,c(-0.5,-0.01),c(0.5,0.01)) 
    tmp_mean[i] = proposal[1]
    tmp_precision[i] = proposal[2]
    
    ABC_result = ABC_acceptance(proposal,threshold,summary,metric)
    
    if(ABC_result[1]== T){
      chain[i+1,] = proposal
      sim_acceptance[i]=T
      sim_mean[i] = ABC_result[2]
      sim_precision[i] = ABC_result[3]
    }else{
      chain[i+1,] = chain[i,]
      sim_acceptance[i]=F
      sim_mean[i] = ABC_result[2]
      sim_precision[i] = ABC_result[3]
    }
  }
  
  sim = matrix(0,ncol=5,nrow=iters)
  sim_mean = data.frame(sim_mean)
  sim_mean = t(sim_mean)
  sim_precision = data.frame(sim_precision)
  sim_precision = t(sim_precision)
  
  sim = data.frame(index = c(1:iters),tmp_mean,tmp_precision,sim_mean,sim_precision,sim_acceptance)
  return(sim)
}


#######################################################################################################################

run_ABC <- function(input) {
  if (input$method == "Rejection") {
    return(run_ABC_rejection(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,input$iterations, input$threshold, input$metric, input$summary))
  } else if (input$method == "ABC_MCMC") {
    return(run_MCMC_ABC(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,input$iterations, input$threshold, input$metric, input$summary))
  } else if (input$method == "ABC_SMC") {
    return(run_ABC_SMC(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,input$iterations, N, tau,input$metric,input$summary))
  }
}



#########################################################################################################################
normal_plot <- function(mean, sd){
  gnorm <- ggplot(NULL,aes(x=c(mean - 4*sd,mean+4*sd))) +
    stat_function(fun=dnorm, args=list(mean=mean, sd=sd)) +
    stat_function(fun=dnorm, args=list(mean=mean, sd=sd), geom="ribbon", fill="gold1", alpha=0.5, mapping = aes(ymin=0, ymax=..y..)) +
    labs(x="p", y="pdf")
  gnorm
}

gamma_plot <- function(s, r){
  ggamma <- ggplot(NULL,aes(x=c(0, 10))) +
    stat_function(fun=dgamma, args=list(shape=s, rate=r)) +
    stat_function(fun=dgamma, args=list(shape=s, rate=r), geom="ribbon", fill="gold1", alpha=0.5, mapping = aes(ymin=0, ymax=..y..)) +
    labs(x="p", y="pdf")
  ggamma
}

mean_trace_plot<-function(ourdata) {
  ggplot(ourdata,aes(x=index, y=sim_mean)) +
    geom_point(data=filter(ourdata, sim_acceptance==FALSE), aes(x=index, y=sim_mean, color="red")) +
    geom_point(data=filter(ourdata, sim_acceptance==TRUE)) + 
    geom_line(data=filter(ourdata, sim_acceptance==TRUE)) +
    ylim(c(50,100))     
}

precision_trace_plot<-function(ourdata) {
  ggplot(ourdata,aes(x=index, y=sim_precision)) +
    geom_point(data=filter(ourdata, sim_acceptance==FALSE), aes(x=index, y=sim_precision, color="red")) +
    geom_point(data=filter(ourdata, sim_acceptance==TRUE)) + 
    geom_line(data=filter(ourdata, sim_acceptance==TRUE))
}

smc_mean_posterior_plot<-function(ourdata){
  ggplot(ourdata,aes(x = saved_mean)) +
    geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.5) 
}

smc_precision_posterior_plot<-function(ourdata){
  ggplot(ourdata,aes(x = saved_precision)) +
    geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.5) 
}  

#############################################################################################################3
shinyServer(function(input, output) {

  output$priorPrecisionPdf <- renderPlot({
    gamma_plot(s=input$sPrior, r=input$rPrior) + lims(y=c(0, input$ymax2))
  })
  
  output$priorMeanPdf <- renderPlot({
    normal_plot(mean=input$meanPrior, sd=input$sdPrior) + lims(y=c(0,input$ymax1))
  })
  
  simulation = eventReactive(input$runButton, {
    ABC_rejection_simulation = run_ABC(input)
  })
  
  output$traceMean <- renderPlot({
    if (input$method == "ABC_SMC"){
      print("no trace polt available")
    } else {
      mean_trace_plot(simulation())
    }
  })
  
  output$tracePrecision <- renderPlot({
    if (input$method == "ABC_SMC"){
      print("no trace polt available")
    } else {
    precision_trace_plot(simulation())
    }
  })
  
  output$posteriorMeanPdf <- renderPlot({
    simulation() %>% filter(sim_acceptance == TRUE) %>%
      ggplot(aes(x = tmp_mean)) +
      geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.5)
  })
  
  output$posteriorPrecisionPdf <- renderPlot({
    simulation() %>% filter(sim_acceptance == TRUE) %>%
      ggplot(aes(x = tmp_precision)) +
      geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.02)
  })
})
