library(shiny)

shinyUI(fluidPage(
  
  h2("Approximate Bayesian Computation"),
  h4(tags$i("Shuni Li, Yueyi Kate Li, Weifang Liu")),
  h4(tags$i("Bayesian Statistics, December 19, 2017")),
  
  hr(),
  
  h3("Introduction"),
  p('One of the basic problems in Bayesian inference is to compute posterior distributions for a given model.
    Suppose data \\(D\\) is generated from a model determined by parameter \\(\\theta\\) and the prior of \\(\\theta\\) is denoted by \\(\\pi(\\theta)\\). The posterior distribution of interest is \\(f(\\theta|D)\\), which is given by'),
  helpText('$$f(\\theta | D) \\propto f(D|\\theta) \\pi(\\theta).$$'),
  p('Common posterior approximation methods such as MCMC and Gibbs Sampling depend on knowing a likelihood function \\(f(D|\\theta)\\). 
    However, for many complex probability models, such likelihoods are either analytically unavailable or computationally difficult. 
    Approximate Bayesian Computation (ABC), also known as likelihood-free computation, effectively circumvents this issue by making use of comparisons between simulated and observed summary statistics.'),
  p('In general, there are three basic ABC algorithms: ABC rejection, ABC Markov Chain Monte Carlo (ABC-MCMC) and ABC Sequential Monte Carlo (ABC-SMC). 
    In this report, we will explain the general procedures for all three methods and demonstrate their usefulness on a Markov Multi-state model. 
    Then, we show a Shiny APP which illustrates the ABC methods on a simpler Normal-Normal model. 
    We conclude with discussion on siginificance of ABC parameters, such as summary statistics, metrics and threshold. '),
  
  hr(),
  
  h3("Model and Data"),  
  p("Multi-state models are stochastic processes models with discrete state space. A Markov multi-state model assumes that 
    future states depend only on the current state, which is the Markov property. 
    More details about the model can be found in", tags$i("Statistical Models"), "(Davison, 2009)."),
  
  p("Considerable computational difficulties arise for inference of continuous time multi-state models when the process is 
    only observed at discrete time points (Tancredi, 2013). For general multi-state Markov model, the evaluation of the likelihood function needs intensive numerical approximations. 
    Moreover, transitions between states may depend on the time since entry into the current state, 
    therefore the likelihood function may not be available in closed form. For the next section of this report, 
    we will apply ABC methods to approximate posterior for a Markov Multi-state model based on a breast cancer data that comes from de Stavola (1988)."),
  
  p("The data is on 37 women with breast cancer treated for spinal metastases at the London Hospital (de Stavola, 1988). Their ambulatory status - defined as ability to walk unaided or not 
    was recorded before treatment began, as it started, and then 3, 6, 12, 24, and 60 months after treatment. The three states are: able to walk unaided (1); unable to walk unaided (2); 
    and dead (3). For example, a sequence 111113 means that the patient was able to walk unaided each time she was seen, 
    but was dead five years after the treatment began. Below is a table of the dataset (Davison, 2009)."), 
  
  fluidRow(
    column(3, p("")),
    column(6,
    tags$img(src='data.png', width = "500px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
    p("Table 1: The table gives the initial and follow-up status for 37 breast cancer patients treated for spinal metastases. 
      The status is able to walk unaided (1), unable to walk unaided (2), or dead (3), and the times of follow-up are 0, 3, 6, 12, 24, and 60 months after treatment began. 
      Woman 24 was alive after 6 months but her ability to walk was not recorded (Davison, 2009).")),
    column(3,p(""))
  ),
  
  p("The parameters in our model
    are \\(\\gamma_{rs}\\)s, where \\(\\gamma_{rs}\\) is the transition density from state \\(r\\) to state \\(s\\). Since state 3
    stands for death, we know it is an absorbing state. Therefore, we have six parameters to estimate, namely \\(\\gamma_{11}\\), \\(\\gamma_{12}\\), \\(\\gamma_{13}\\),
    \\(\\gamma_{21}\\),\\(\\gamma_{22}\\), and \\(\\gamma_{23}\\)."),
  
  p("Our priors for \\(\\gamma_{rs}\\)s are modeled by beta distributions, where the random variables can take continuous values from 0 to 1."),
  
  p("Since the likelihood function is intractable when we only have observations at discrete time points, we estimate the likelihood by the 
    transition probabilities. For example, the probability that a patient at state 1 goes to state 2 at the next time point is given by
    \\(\\gamma_{12}/(\\gamma_{11}+\\gamma_{12}+\\gamma_{13})\\)."),
  
  
  hr(),
  
  h3("ABC in Action"),
  
  p("Althogh there are different ABC algorithms, they all incorporate a common procedure to obtain a random sample from the posterior distribution. 
    For a candidate parameter value \\(\\theta\\) drawn from some density, a simulated data set \\(D\\) is generated from the likelihood function \\(f(D|\\theta)\\). 
    This candidate is then accepted if simulated and observed data are sufficiently “close”. 
    Here, closeness is measured by the distance between two vectors of summary statistics \\(S_{obs}\\) and \\(S_{sim}\\). The distance is calculated according to a metric function \\(\\rho\\).
    That is , if \\(\\rho( S_{obs}, S_{sim}) \\leq \\epsilon\\), where \\(\\epsilon\\) is a fixed threshold, the algorithm accepts this candidate (Sisson et al, 2007)."),
  
  p("Now we are ready to see how this simple idea is combined with other techiniques to approximate posteriors."),
  
  h4("ABC Rejection"),
  p(" ABC rejection algorithm is very intuitive. Essentially, we simulate many, many datasets from a given model. 
    If a simulated dataset is very similar or close to the observed dataset, then we accept this dataset.
    The reasoning behind is method is simple: the probability that the simulated data is close to the observed data is approximately proportional to it being identical to the data.
    Thus, the parameter values obtained from accepted datasets will well approximate the interested posterior.
    
    The general procedures for ABC rejection are as follows:",
    tags$ol(tags$li("Compute summary statistics for the observed data"),
            tags$li("Draw parameter values from the prior distribution"),
            tags$li("Simulate data for each parameter drew in step two and compute summary statistics for each simulated data"),
            tags$li("Accept or reject this data based on its deviation from the observed data"))),
  
  p("Notice that the acceptance rate for ABC Rejection algorithm can be very low, since candidate parameter values are generated from the prior \\(\\pi(\\theta)\\), which may be diffuse with respect to the posterior.
    To obtain a good approximation, we might need to perform much more iterations than other methods."),
 
  hr(),
  
  p("Now we sill illustrate how ABC Rejection works using the breast cancer dataset."),
  p("We will use mean and standard deviation as summary statitics, and euclidean distance as metric for the following model. 
     There might be better choices, but these are the most common ones to start. We have a more detailed discussion for summary statistcis and metric in the later section."),
  
  p("First, we compute the summary statistics for
the observed dataset. Let \\(n_{ij}\\) be the number of transitions from state \\(i\\) to state \\(j\\). From the dataset above, we can see that \\(n_{11} = 38\\), \\(n_{12} = 12\\), \\(n_{13} = 13\\), \\(n_{21} = 2\\), \\(n_{22} = 14\\),\\(n_{23} = 19\\), which means
    that we observed \\(38\\) transitions from state \\(1\\) to state \\(1\\), \\(12\\) transitions from state \\(1\\) to state \\(2\\), and so on."),
  
  p("Our prior distributions for \\(\\gamma_{rs}\\)s are Beta distributions with \\(\\alpha=1\\) and \\(\\beta=10\\). We use Beta distributions because we want the transition density to be
    small but positive.
    From each set of generated \\(\\gamma_{rs}\\)s, we estimated the likelihood by normalizing each \\(\\gamma_{rs}\\) as mentioned at the end of 
    the last section. With the estimated likelihoods, we can generate data for the 37 patients with their initial state for a
    60 month period. Finally, we compute the summary statistics for each of the generated dataset and accept the set of \\(\\gamma_{rs}\\)s if the distance
    between the summary statistics between the observed dataset and the generated dataset is smaller than a given threshold.
    The histograms below show the marginal distributions of the accepted values for the parameters with their prior distributions (black) and posterior means (red).
    "),
  
  fluidRow(
    column(5, tags$img(src='rej1.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 1: ABC Rejection: \\(\\gamma_{11}\\)", align = "center")),
    column(5,
           tags$img(src='rej2.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 2: ABC Rejection: \\(\\gamma_{12}\\)", align = "center"))
  ),
  
  fluidRow(
    column(5, tags$img(src='rej3.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 3: ABC Rejection: \\(\\gamma_{13}\\)", align = "center")),
    column(5, tags$img(src='rej4.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 4: ABC Rejection: \\(\\gamma_{21}\\)", align = "center"))
  ),
  
  fluidRow(
    column(5, tags$img(src='rej5.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 5: ABC Rejection: \\(\\gamma_{22}\\)", align = "center")),
    column(5, tags$img(src='rej6.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 6: ABC Rejection: \\(\\gamma_{23}\\)", align = "center"))
  ),
  
  p("From the posterior means, we are able to estimate the transition probabilities. For example, the transition probability from state 1 to state 2
    is 30%, from state 2 to state 1 is 18%, and from state 2 to state 3 is 57%. The results are not very accurate compared to the results in Tancredi (2013), but
    they are consistent with the observed summary statistics in general, and it makes sense that it is more likely to enter state 2 (unable to walk unaided) from state 1 (walk unaided) than from state 1 to state 2, 
    and it is likely that a person in state 2 goes to state 3 (death) in 60 months. Also, due to computation limit, we set the number of iterations as 5000. More iterations might lead to better results. 
    "),
  

  h4("ABC-MCMC"),
  p("To overcome the inefficieny of the simple rejection algorithm, Marjoram et al. proposed to embed the rejection algorithm within the well known MCMC framework to reduce its complexity (Marjoram et al., 2003). 
    MCMC does a random walk (Markov-chain) in parameter space, and thereby concentrates sampling on the important parameter areas. The general procedures for ABC-MCMC are as follows:",
    tags$ol(tags$li("Suppose we are currently at \\(\\theta_i\\)"),
            tags$li("Generate a candidate value \\(\\theta_{i+1}\\) according to some symmetric proposal distribution \\(q(θ_{i+1} | θ_i)\\)"),
            tags$li("Simulate a dataset using \\(\\theta_{i+1}\\) and compute its summary statistics"),
            tags$li("If simulated data is close to observed data, go to next step, and otherwise stay at \\(\theta_i\\) and return to step 1"),
            tags$li("Compute \\(h = h(\\theta_{i}, \\theta_{i+1}) =  \\text{min}(1, \\frac{\\pi(\\theta_{i+1})q(\\theta_{i+1} \\rightarrow \\theta_{i})}{\\pi(\\theta_{i})q(\\theta \\rightarrow \\theta_i)})\\)"),
            tags$li("Move to \\(\\theta_{i+1}\\) with probability \\(h\\), and otherwise stay at \\(\\theta\\), then return to the first step with \\(\\theta_{i} = \\theta_{i+1}\\)"))),
  
  p("ABC-MCMC Algorithm generates a sequence of highly correlated samples from \\(f(\\theta)|\\rho(S(x),S(x_{0})) \\leq \\epsilon)\\).
    Notice that the likelihood ratio in the normal Metropolis Hasting algorithm is now coarsely approximated by 1 if simulated and observed data are sufficiently “close”, and 0 otherwise.
    The stationary distribution of the resulting chain is indeed \\(f(\\theta|D)\\). The proof can be found in the paper", tags$i("Markov chain Monte Carlo without likelihoods"), "by Marjoram et al.    
    The chain length, \\(N\\), can be obtained through a careful assessment of convergence."),
  
  p("When the prior and posterior are dissimilar, Majoram et al. report that algorithm ABC-MCMC delivers substantial increases in acceptance rates over algorithm ABC rejection
    although at the price of generating dependent samples. However, if the ABC-MCMC enters an area of relatively low probability with a poor proposal mechanism, 
    the efficiency of the algorithm is strongly reduced as it then becomes difficult to move anywhere with a reasonable chance of acceptance, 
    and so the samples get stuck in that part of the state space for long periods of time."),
  
  hr(),
  p("Now we apply ABC-MCMC to the same example and compare its performanace with the rejection algorithm.
    We use the same prior distributions as in ABC rejection and run 2000 simulations this time. The distance is computed as the Euclidean distance
    between the summary statistics of the observed data and generated data. The threshold we use to obtain the following results is 40. Again, the histrograms below show the marginal distributions of the accepted
    parameters and their prior distributions.
    "),
  
  fluidRow(
    column(5, tags$img(src='MCMC1.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 7: ABC-MCMC: \\(\\gamma_{11}\\)", align = "center")),
    column(5,
           tags$img(src='MCMC2.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 8: ABC-MCMC: \\(\\gamma_{12}\\)", align = "center"))
  ),
  
  fluidRow(
    column(5, tags$img(src='MCMC3.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figue 9: ABC-MCMC: \\(\\gamma_{13}\\)"), align = "center"),
    column(5, tags$img(src='MCMC4.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
             p("Figue 10: ABC-MCMC: \\(\\gamma_{21}\\)", align = "center"))
  ),
  
  fluidRow(
    column(5, tags$img(src='MCMC5.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
             p("Figue 11: ABC-MCMC: \\(\\gamma_{22}\\)", align = "center")),
    column(5, tags$img(src='MCMC6.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
             p("Figue 12: ABC-MCMC: \\(\\gamma_{23}\\)", align = "center"))
  ),
  
  p("Compare to ABC Rejection, ABC-MCMC yields results more consistent with Tancredi (2013) and Davison (2009). 
    The transition probability from state 1 to state 1 is 57%, from state 1 to state 2 is 20%, from state 2 to state 1 is 18%, from state 2 to state 2 is 24%, and from state 2 to state 3 is 58%.
    The probabilities also agree with the observed summary statistics in general.
    "),

  hr(),
  
  h4("ABC-SMC"),
  p('Both rejection sampling or ABC-MCMC methods can be highly inefficient, and accordingly require far more iterations than may be practical to implement. The Sequential Monte Carlo (SMC) method overcomes these inefficiencies.'),
  p('ABC-SMC is a particle filter approach, that starts by first generating \\(N\\) particles from the prior.
    In the following iterations, the algorithm selects a particle from the obtained distribution, perturbates it, and then simulates data
    using that parameter combination. If the obtained data is sufficiently similar to the observed data that we are trying to fit on,
    the parameter combination is accepted. As soon as \\(N\\) parameter combinations are accepted, these replace the prior,
    and the algorithm starts anew. As the number of iterations increases, the acceptance threshold approaches zero,
    such that the simulated data needs to be more and more similar to the observed data (Sisson et al, 2017).'),
  p("Given a decreasing sequence of tolerance threshold \\(\\epsilon_1, \\cdots, \\epsilon_T\\) and the prior distribution\\(\\pi(\\theta)\\), the general ABC-SMC procedures are as follows:",
    tags$ol(tags$li("At iteration \\(t = 1\\)"), tags$ol(tags$li("Draw parameter values from the prior distribution"),
                                                         tags$li("Simulate data for each parameter drew in step two"),
                                                         tags$li("Repeat until we found an accepted dataset"),
                                                         tags$li("Set weights \\(\\omega^{1}_{i} = 1/N\\)")),
            tags$li("At iterations \\(2 \\leq t \\leq T\\)",
                    tags$ol(tags$li("Draw parameter value \\(\\theta_i\\) with probabilities \\(\\omega^{t-1}_j\\)"),
                            tags$li("Simulate data for each parameter drew in step two"),
                            tags$li("Repeat until we found an accepted dataset"),
                            tags$li("Set weights \\(\\omega^{t}_{j} \\propto \\frac{\\pi(\\mu^t)}{\\sum_{j=1}^N w_j^{t-1}N(\\mu_i^t;\\mu_j^{t-1},\\sigma^2 t)} \\)"))))),
hr(),  
p("We apply ABC-SMC to the breast caner dataset. We use \\(beta(2,2),beta(2,4),beta(2,5),beta(1,15),beta(2,4),beta(2,4)\\) as the priors
    for \\(\\gamma_{11}\\), \\(\\gamma_{12}\\), \\(\\gamma_{13}\\),\\(\\gamma_{21}\\),\\(\\gamma_{22}\\),\\(\\gamma_{23}\\). The distance function
    is \\(\\sum \\frac{(n_{rs}(x)-n_{rs}(z))^2}{n_{rs}(x))},\\text{ where } n_{rs}(x)\\) represents the number of transitions from \\(r\\) to \\(s\\)
    for the observed data and \\(n_{rs}(x)\\) represents the that for the sample data. After simulating
    the process for three rounds with 1000 sampled data and decreasing thresholds \\((500,400,120)\\) for each round, we get the following posterior distribution
    for each \\(\\gamma\\).
    "),
  
  fluidRow(
    column(5, tags$img(src='SMC1.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figure 13: ABC-SMC: \\(\\gamma_{11}\\)", align = "center")),
    column(5,
           tags$img(src='SMC2.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figure 14: ABC-SMC: \\(\\gamma_{12}\\)", align = "center"))
  ),
  
  fluidRow(
    column(5, tags$img(src='SMC3.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figure 15: ABC-SMC: \\(\\gamma_{13}\\)"), align = "center"),
    column(5, tags$img(src='SMC4.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figure 16: ABC-SMC: \\(\\gamma_{21}\\)", align = "center"))
  ),
  
  fluidRow(
    column(5, tags$img(src='SMC5.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figure 17: ABC-SMC: \\(\\gamma_{22}\\)", align = "center")),
    column(5, tags$img(src='SMC6.png', width = "400px", align = "center", style="display: block; margin-left: auto; margin-right: auto;"),
           p("Figure 18: ABC-SMC: \\(\\gamma_{23}\\)", align = "center"))
  ),
  
  p("Our results show that the probability of staying in state 1 is 56.23%, from state 1 to 2 is 27.42%, from 2 to 1 is 36.93%, from state 2 to 2 is 35.61%, from state
    2 to 3 is 27.44%. The probability of transitioning from 2 to 1 seems a little bit high compared to the observed data. This posterior approximation
    given by ABC-SMC using the number of transitions as summary statistics is not as good as that given by ABC-MCMC using euclidean distance."),
  
  # p("Comparing the running time between ABC-SMC and ABC-Rejection, 
  #  we found that they gave a fairly good estimation for the mean, but nor for the variance, 
  #  which we will investigate further. We also found that ABC-SMC is more efficient compared to ABC-Rejection, 
  #  as expected. The time is almost cut by half."),
  
    hr(),

  
  h3("Shiny Visulization"),
  
  p(tags$i("If you had an error while using the Shiny APP, please make sure that you specify all parameters and the parameters are in reasonable range (especially the threshold.")),

  h4("The Story"),
  withMathJax(),
  p("Given a student’s scores on the first five Bayes homeworks, we can construct a normal-normal Bayesian model to make inferences about this student’s underlying Bayes skill level:"), 
  helpText('$$X_i | \\mu, \\sigma^2 \\sim N(\\mu, \\tau^{-1})$$'),
  helpText('$$\\mu \\sim N(85, 3^2)$$'),
  helpText('$$\\tau \\sim Gamma(3, 5)$$'),
  
  hr(),
  
  h4("Tune Prior Parameters"),
  withMathJax(),
  p('Recall that \\(N(\\mu, \\tau)\\) models the average bayes score of this student. Use the slider to update \\(\\mu \\sim N(85, 9)\\). This particular normal prior indicates that his score is likely between 74 and 94. Use the third slider to adjust y-aixs limit of the plot.'),
  
  
  fluidRow(
    column(4,
           #h4("Tune the Normal(mu,sd) prior:"), 
           sliderInput("meanPrior", "parameter mu", min = 0, max = 100, value = 85),
           sliderInput("sdPrior", "parameter sd", min = 0, max = 100, value = 3),
           sliderInput("ymax1", "Specify the limit of the y-axis:", min = 0, max = 1.5, value = 0.25)
    ),
    
    column(6,
           h4("Plot 1: Prior mean pdf:"), 
           plotOutput("priorMeanPdf", width = "450px", height = "200px")
    )
  ),
  
  hr(),

  p("The parameter tau models the precision of scores. Update it to \\(Gamma(3, 5)\\), which indicates that variance of scores is 0-10 from homework to homework. Use the third slider to adjust y-aixs limit of the plot."),
  
  fluidRow(
    column(4,
           # the Gamma prior
           #h4("Tune the Gamma(s,r) prior:"), 
           sliderInput("sPrior", "parameter s", min = 0, max = 100, value = 3),
           sliderInput("rPrior", "parameter r", min = 0, max = 100, value = 5),
           sliderInput("ymax2", "Specify the limit of the y-axis:", min = 0, max = 2, value = 1.5)
    ),
    
    column(6,
           h4("Plot 2: prior pdf for precision:"), 
           plotOutput("priorPrecisionPdf", width = "450px", height = "200px")
    )
  ),
  
  hr(),
  
  h4("Set ABC parameters & Run ABC"),
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
      
      sliderInput("iterations", "Specify number of iterations:", value = 5000, min = 1000, max = 10000, step= 100),
      
      sliderInput("threshold", "Specify threshhold for \\(\\mu\\):", min = 0, max = 30, value = 10, step = 1),
      
      #sliderInput("tauthreshold", "Specify threshhold for \\(\\tau\\):", min = 0, max = 30, value = 10, step = 1),
      
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
  ),
  
  
  h3("Discussion"),
  h4("Summary Statistics and Metric"),
  
  p("The choice of the summary statistics are paramount to the success of the approximation.
    We want to use summary statistics that reduces the dimensionality of the data without loosing information. However, there are fundamental difficulty associated with the choice of the summary statistic: 
    it is usually impossible to find non-trivial sufficient statistics and the selection is highly empirical. In the breast cancer example, euclidean distance
    gives more accurate approximation. In the normal-normal model, meam and standard deviation 
    combined capture the data well. Adding range and median does not add much more information to the model. For metric, L-2 norm gives slightly better posterior
    approximation than L-1 norm, while L-1 norm is more efficient."),
  
  h4("Prior distributions"),
  
  p("The performance of the three ABC algorithms are dependent on prior knowldege. For ABC rejection, if the prior and the posterior distributions are very
    different, the acceptance rate can be very low, resulting in large number of simulations. ABC-MCMC and ABC-SMC try to tackle this problem. ABC-MCMC generates more parameters in the area of high
    likelihood of being accepted. However, if it gets stuck, it will result in a inefficient long chain. ABC-SMC assigns higher weights to the simulated parameters that are not sampled from the prior. As round increases, the distribution
    of the accepted parameters evolve gradually from the prior distribution towards the target distribution.
    "),
  
  
  h4("Threshold"),
  
  p("Through our simulation study, we find that threshold has a substantial impact on the accuracy of the posterior approximations. If the threshold is too small, then
    it is hard to draw parameters that produce similar summarty statistics to that of the observed data. If the threshold is too large,
    then our approximation becomes inaccurate. Moreover, the inexactness introduced by the distance may produce a bias in the posterior distribution. More simulation study is needed
    to understand the sensitivity of the posterior distribution to threshold."),
  
  h4("Number of iterations"),
  
  p("Larger number of iterations does not guarantee a better posterior approximation. This is because that the algorithm may suffer
    from nonsufficient summary statistics, misspecified models and priors, inappropriate threshold, and other problems that can potentially
    introduce a bias in the results. However, with reasonable prior and other parameters, more iterations lead to more stable approximations. "),
  
  h3('Conclusion and Future Work'),
  
  p("In this project, we have studied three ABC algorithms: ABC rejection, ABC-MCMC and ABC-SMC. Using the Markov model based on breast cancer data,
    we demonstrate the usefulness of all three algorithms in approximating poteriors when the likelihood function is unavailable. 
    For this particular model, we find that the ABC-MCMC is the most effective, while ABC-SMC performs poorly.
    We need to tune the priors to actually get a closer approximation for the SMC algorithm. 
    In addition, the Shiny APP we implemented for Normal-Normal model serves more like an educational tool for
    interested readers to carry out various simulation studies. This app also helps us yo gain better intuition towards ABC parameters, such as threshild, summay statistics, 
    metric etc. 

There is a lot we can do for future work. For example, we can build more sophisticated algorithms with better distance functions,and investigate how the factors we
    mentioned above impact the accuracy and efficiency of the algorithms. We can also compare ABC to other simulation-based algorithms. Since we only implement ABC on a couple of examples, it would
    also be interesting to see how ABC performs for other complex models."),
  
  h3('References'),
  
  tags$ol(tags$li("Marjoram, P., Molitor, J., Plagnol, V., & Tavare, S. (2003). Markov chain Monte Carlo without likelihoods. ", tags$i("Proceedings of the National Academy of Sciences, 100"), "(26), 15324-15328. doi:10.1073/pnas.0306899100"),
          tags$li("Davison, A. C. (2009). ", tags$i("Statistical models."), "Cambridge: Cambridge University Press."),
          tags$li("Tancredi, A. (2013). Approximate bayesian Inference for discretely observed continuous- time multi-state models. ", tags$i("In Proceedings of the Conference SCO 2013")),
          tags$li("Cowles, M. K., & Carlin, B. P. (1996). Markov Chain Monte Carlo Convergence Diagnostics: A Comparative Review.", tags$i("Journal of the American Statistical Association, 91","(434), 883. doi:10.2307/2291683.")),
          tags$li("Sisson, S. A., Fan, Y., & Tanaka, M. M. (2007). Sequential Monte Carlo without likelihoods.", tags$i("Proceedings of the National Academy of Sciences, 104", "(6), 1760-1765. doi:10.1073/pnas.0607208104")))

))