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

run_ABC <- function(input) {
  if (input$method == "Rejection") {
    return(run_ABC_rejection(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,input$iterations, input$threshold, input$metric, input$summary))
  } else if (input$method == "ABC_MCMC") {
    # TODO: replace with ABC MCMC
  } else if (input$method == "ABC_SMC") {
    # TODO: replace with ABC SMC
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
    geom_point(data=filter(ourdata, sim_acceptance==TRUE)) + 
    geom_line(data=filter(ourdata, sim_acceptance==TRUE)) + 
    geom_point(data=filter(ourdata, sim_acceptance==FALSE), aes(x=index, y=sim_mean, color="red")) +
    ylim(c(50,100))     
}

precision_trace_plot<-function(ourdata) {
  ggplot(ourdata,aes(x=index, y=sim_precision)) +
    geom_point(data=filter(ourdata, sim_acceptance==TRUE)) + 
    geom_line(data=filter(ourdata, sim_acceptance==TRUE)) + 
    geom_point(data=filter(ourdata, sim_acceptance==FALSE), aes(x=index, y=sim_precision, color="red"))
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

  output$text2 <- renderUI({
    HTML("my awesome text message in HTML!!!")
  })
  
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
    mean_trace_plot(simulation())
  })
  
  output$tracePrecision <- renderPlot({
    precision_trace_plot(simulation())
  })
  
  output$posteriorMeanPdf <- renderPlot({
    simulation() %>% filter(sim_acceptance == TRUE) %>%
      ggplot(aes(x = tmp_mean)) +
      geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.5)
  })
  
  output$posteriorPrecisionPdf <- renderPlot({
    simulation() %>% filter(sim_acceptance == TRUE) %>%
      ggplot(aes(x = tmp_precision)) +
      geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.1)
  })
})
