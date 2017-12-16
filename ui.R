library(shiny)

shinyUI(fluidPage(
  
  h2("Approximate Bayesian Computation on Normal-Normal"),
  
  hr(),
  
  h3("Section I: Normal-Normal model"),
  
  
  p("Given a student’s scores on the first five Bayes homeworks, we can construct a normal-normal Bayesian model to make inferences about this student’s underlying Bayes skill level"), 
  p("How to add Latex form equations????"),
  hr(),
  
  h3("Section II: Tune Prior Parameters"),
  
  p("Recall that N(mu,sd) models the average bayes score of this student. Use the slider to update mu~N(85,9). This particular normal prior indicates that his score is likely between 74 and 94. Use the third slider to adjust y-aixs limit of the plot."),
  
  fluidRow(
    column(4,
           #h4("Tune the Normal(mu,sd) prior:"), 
           sliderInput("meanPrior", "parameter mu", min = 0, max = 100, value = 5),
           sliderInput("sdPrior", "parameter sd", min = 0, max = 100, value = 5),
           sliderInput("ymax1", "Specify the limit of the y-axis:", min = 0, max = 1.5, value = 0.1)
    ),
    
    column(6,
           h4("Plot 1: Prior mean pdf:"), 
           plotOutput("priorMeanPdf", width = "450px", height = "200px")
    )
  ),
  
  hr(),

  p("The parameter tau models the precision of scores. Update it to Gamma(3, 5), which indicates that variance of scores is 0-10 from homework to homework. Use the third slider to adjust y-aixs limit of the plot."),
  
  fluidRow(
    column(4,
           # the Gamma prior
           #h4("Tune the Gamma(s,r) prior:"), 
           sliderInput("sPrior", "parameter s", min = 0, max = 100, value = 5),
           sliderInput("rPrior", "parameter r", min = 0, max = 100, value = 5),
           sliderInput("ymax2", "Specify the limit of the y-axis:", min = 0, max = 1.5, value = 0.1)
    ),
    
    column(6,
           h4("Plot 2: prior pdf for precision:"), 
           plotOutput("priorPrecisionPdf", width = "450px", height = "200px")
    )
  ),
  
  hr(),
  
  h3("Section II: Set ABC parameters"),
  p("The ABC algorithms dependend hevealy on parameters, such as the number of iterations, rejection threshold, and summary statistic methods. After setting these parameters, click run to produce trace plots and approximated posteriors."),
  
  sidebarLayout(
    sidebarPanel(
      selectInput("method", "Specify ABC method:",
                  c("Rejection" = "Rejection",
                    "ABC_MCMC" = "ABC_MCMC",
                    "ABC_SMC" = "ABC_SMC")),
      
      checkboxGroupInput("summary", "Choose the summary statistics",
                         c("Mean" = "mean", 
                           "Standard Deviation" = "sd",
                           "Median" = "median",
                           "Range" = "range",
                           "Just data" = "data")),
      
      
      selectInput("metric", "Choose the metric:",
                  c("L-1 Norm" = "L1Norm",
                    "L-2 Norm" = "L2Norm")),
      
      # the Gamma prior
      #h4("Tune the Gamma(r,s) prior:"), 
      #sliderInput("rPrior", "parameter r", min = 0, max = 100, value = 5),
      #sliderInput("sPrior", "parameter s", min = 0, max = 100, value = 5),
      
      # the Normal prior 
      #h4("Tune the Normal(mu,sd) prior:"), 
      #sliderInput("meanPrior", "parameter mu", min = 0, max = 100, value = 5),
      #sliderInput("sdPrior", "parameter sd", min = 0, max = 100, value = 5),
      
  
      #sliderInput("ymax", "Specify the limit of the y-axis:", min = 0, max = 1, value = 0.1),
      
      sliderInput("iterations", "Specify number of iterations:", value = 5000, min = 1000, max = 10000, step= 100),
      
      sliderInput("threshold", "Specify threshhold:", min = 0, max = 100, value = 30, step = 5),
      
      actionButton("runButton", "Run Simulation")
    ),
    
    mainPanel(
    
      #h4("Plot of the prior pdf for mean:"), 
      #plotOutput("priorMeanPdf", width = "450px", height = "200px"),
      #h4("Plot of the prior pdf for precision:"), 
      #plotOutput("priorPrecisionPdf", width = "450px", height = "200px"),
      
      h4("Plot 3: proposed mean"), 
      plotOutput("traceMean", width = "500px", height = "250px"),
      h4("Plot 4: proposed precision"), 
      plotOutput("tracePrecision", width = "500px", height = "250px"),
      
      h4("Plot 5: posterier mean"), 
      plotOutput("posteriorMeanPdf", width = "450px", height = "200px"),
      h4("Plot 6: posterier precision"), 
      plotOutput("posteriorPrecisionPdf", width = "450px", height = "200px")
    )
  )
))