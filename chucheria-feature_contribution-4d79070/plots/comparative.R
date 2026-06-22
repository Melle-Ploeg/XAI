library(dplyr)
library(ggplot2)
library(hrbrthemes)

# colors
#["#04A0A7",
#    "#5fcfd2",
#    "#6788ac",
#    "#4584ff",
#    "#964deb"
# neutral #C8C8C8]

import_roboto_condensed()
new_theme <- theme_ipsum_rc(base_size = 20, plot_title_size = 26,
                            plot_title_margin = 15, subtitle_size = 20,
                            axis_title_size = 15)
theme_set(new_theme)


data <- function(file) {
  data <- readr::read_csv(file) %>% 
    group_by(method) %>% 
    summarize(across(everything(), mean)) %>% 
    tidyr::pivot_longer(!c(method, obs))
  return(data)
}

plot <- function(df, title, subtitle, legend, filename) {
  ggplot(df, aes(x = as.factor(name), 
                 y = value, 
                 fill = factor(method, 
                               levels=c('contribution', 'shap', 'lime'))
                 )
         ) +
    geom_bar(stat='identity', position='dodge') + 
    labs(title = title,
         subtitle = subtitle) +
    scale_fill_manual(values = c( "#04A0A7", "#964deb", "#C8C8C8")) +
    ylab("Contribution") +
    xlab("Feature") +
    guides(fill = guide_legend(title = legend)) +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(paste0('data/figures/', filename), device = "png", dpi = 320, 
         width = 40, height = 15, units = "cm")
  
}


files <- list.files(path = "data/output", pattern = 'comparative', full.names = TRUE)
title <- c('Concrete dataset: Contribution explanation of three algoritms',
           'Diabetes dataset: Contribution explanation of three algoritms')
subtitle <- c('Comparative of contribution mean across test dataset')
legend <- c('Method')
filename <- c('comparative_concrete.png', 'comparative_diabetes.png')


for (i in c(1, 2)) {
  df <- data(files[i])
  plot(df, title[i], subtitle, legend, filename[i])
}

