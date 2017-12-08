library(shiny)

shinyUI(fluidPage(
  sidebarLayout(
    sidebarPanel(
      selectInput("method", "Specify ABC method:",
                  c("Rejection" = "rejection",
                    "Method2" = "m2",
                    "Method3" = "m3")),

      # the Gamma prior
      h4("Tune the Gamma(r,s) prior:"), 
      sliderInput("rPrior", "parameter r", min = 0, max = 100, value = 5),
      sliderInput("sPrior", "parameter s", min = 0, max = 100, value = 5),
      
      # the Normal prior 
      h4("Tune the Normal(mu,sd) prior:"), 
      sliderInput("meanPrior", "parameter mu", min = 0, max = 100, value = 5),
      sliderInput("sdPrior", "parameter sd", min = 0, max = 100, value = 5),
      
      h4("Specify the limit of the y-axis:"), 
      sliderInput("ymax", "", min = 0, max = 1, value = 0.1),
      
      #h4("Specify sample data:"), 
      #sliderInput("n", "n", min = 1, max = 1000, value = 12)
      
      actionButton("runButton", "Run Simulation")
    ),
    
    mainPanel(
      h4("Plot of the prior pdf for mean:"), 
      plotOutput("priorMeanPdf", width = "450px", height = "200px"),
      h4("Plot of the prior pdf for precision:"), 
      plotOutput("priorPrecisionPdf", width = "450px", height = "200px"),
      
      h4("Plot of the proposed mean"), 
      plotOutput("traceMean", width = "500px", height = "250px"),
      h4("Plot of the proposed precision"), 
      plotOutput("tracePrecision", width = "500px", height = "250px"),
      
      h4("Plot of the posterier mean"), 
      plotOutput("posteriorMeanPdf", width = "450px", height = "200px"),
      h4("Plot of the posterier precision"), 
      plotOutput("posteriorPrecisionPdf", width = "450px", height = "200px")
    )
  )
))