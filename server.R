library(ggplot2)
library(dplyr)
library(devtools)
library(coda)
library(shiny)

set.seed(2017)

dat = c(70, 80, 75, 83, 72)
dat = data.frame(dat)
dat = sort(dat[,1])

# Set up number of simulations and number of parameters to save
reps  =  10000
threshold = 30  #Tuning threshold in future
prior_mean = 85
prior_sd = 3
prior_s = 3
prior_r = 5

# The run_ABC_rejection function returns a data table with all simulation records
# Input: 
#       prior_params: prior distrinution parameters
#       n: the total number of simulations
#       threshold: acceptane threshold
# Output: 
#       a data table 'sim' that contains all simulation records
run_ABC_rejection = function(prior_mean, prior_sd, prior_s, prior_r, n, threshold) {
  #sample n pairs of parameters from prior distribution
  tmp_mean <- rnorm(n, mean = prior_mean, sd = prior_sd) #改成class
  tmp_precision <- rgamma(n, prior_s, prior_r)
  
  sim_mean = rep(NA, n)
  sim_precision = rep(NA, n)
  sim_deviation = rep(NA, n)
  sim_acceptance = rep(NA, n)
  
  for (i in 1:n){
    # For each lambda, generate a simulated data with 13 cases
    samples = rnorm(5, tmp_mean[i], sqrt(1/tmp_precision[i]))
    sorted_samples <- sort(samples)
    sim_mean[i] = mean(samples)
    sim_precision[i] = 1/(sd(samples))^2
    
    # compute the deviation by takeing the Euclidean distance between observed summary statistics
    sim_deviation[i] = dist(rbind(dat,sorted_samples))
    #sim_deviation[i] = sqrt(sum((mean(samples) - mean_data)^2 + (sd(samples) - sd_data)^2))
    
    # Accept if deviation is smaller than threshhold; reject otherwise.
    if(sim_deviation[i] < threshold)
      sim_acceptance[i] = TRUE
    else
      sim_acceptance[i] = FALSE
  }
  sim = data.frame(index = c(1:reps), tmp_mean, tmp_precision, sim_mean, sim_precision, sim_deviation, sim_acceptance)
  return(sim)
}

run_ABC <- function(input, threshold) {
  if (input$method == "rejection") {
    return(run_ABC_rejection(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,reps, threshold))
  } else if (input$method == "m2") {
    # TODO: replace with method 2
    print("Method 2 selected")
    return(run_ABC_rejection(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,reps, threshold))
  } else if (input$method == "m3") {
    # TODO: replace with method 3
    print("Method 3 selected")
    return(run_ABC_rejection(input$meanPrior, input$sdPrior, input$sPrior, input$rPrior,reps, threshold))
  }
}

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
    geom_point(data=filter(ourdata, sim_acceptance==FALSE), aes(x=index, y=sim_mean, color="red"))
}

shinyServer(function(input, output) {
  
  output$priorPrecisionPdf <- renderPlot({
    gamma_plot(s=input$sPrior, r=input$rPrior) + lims(y=c(0, 1))
  })
  
  output$priorMeanPdf <- renderPlot({
    normal_plot(mean=input$meanPrior, sd=input$sdPrior) + lims(y=c(0,input$ymax))
  })
  
  simulation = eventReactive(input$runButton, {
    ABC_rejection_simulation = run_ABC(input, threshold)
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
      geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.5) #+
    #geom_vline(xintercept = mean(ABC_rejection_posterior$tmp_mean),color="blue",linetype = "longdash")
    
  })
  
  output$posteriorPrecisionPdf <- renderPlot({
    simulation() %>% filter(sim_acceptance == TRUE) %>%
      ggplot(aes(x = tmp_precision)) +
      geom_histogram(aes(y=..density..), color = 'white', boundary = 0, binwidth = 0.1) #+
    #geom_vline(xintercept = mean(ABC_rejection_posterior$tmp_precision),color="blue",linetype = "longdash")
  })
})