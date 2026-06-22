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
    tidyr::pivot_longer(!col) 
  return(data)
}

plot <- function(df, title, subtitle, legend, filename) {
  ggplot(df, aes(x = as.factor(col), 
                 y = value, 
                 fill = factor(name, 
                               levels=c('contribution', 'shap'))
  )
  ) +
    geom_bar(stat='identity', position='dodge') + 
    labs(title = title,
         subtitle = subtitle) +
    scale_fill_manual(values = c("#04A0A7",  "#964deb", "#C8C8C8")) +
    ylab("Contribution") +
    xlab("Feature") +
    guides(fill = guide_legend(title = legend)) +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  ggsave(paste0('data/figures/', filename), device = "png", dpi = 320, 
         width = 40, height = 15, units = "cm")
  
}


files <- list.files(path = "data/output", pattern = 'outlier', full.names = TRUE)
title <- c('Concrete dataset: Contribution in outlier observation',
           'Diabetes dataset: Contribution in outlier observation')
subtitle <- c('SHAP and Feature contribution comparative')
legend <- c('Method')
filename <- c('outlier_concrete.png', 'outlier_diabetes.png')


for (i in c(1, 2)) {
  df <- data(files[i])
  plot(df, title[i], subtitle, legend, filename[i])
}

