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


data <- function(file, column) {
  data <- readr::read_csv(file) %>% 
    group_by(col) %>% 
    mutate(sum = sum(abs(mean))) %>% 
    filter(sum !=0) %>% 
    ungroup()
    
  return(data)
}

plot <- function(data, order_cols, title, subtitle, legend, filename) { 
  ggplot(data, aes(x = factor(name, levels = order_cols), fill=factor(color))) +
    geom_bar(aes(y = mean), stat='identity') +
    geom_errorbar(aes(ymin=mean-abs(std/2), ymax=mean+abs(std/2)), width=.2) +
    facet_wrap(~ as.factor(col)) +
    labs(title = title,
         subtitle = subtitle) +
    scale_fill_manual(values = c("#04A0A7"), na.value = "#C3C3C3") +
    ylab("Contribution") +
    xlab("Noise level") +
    theme(legend.position="none",
          axis.text.x = element_text(angle = 45, hjust = 1))

  ggsave(paste0('data/figures/', filename), device = "png", dpi = 320, 
         width = 40, height = 30, units = "cm")
}



files <- list.files(path = "data/output", pattern = 'noise', full.names = TRUE)
cols <- c('age', 'bmi')
order_cols <- c("Noise %0", "Noise %100", "Noise %200", "Noise %300", "Noise %400")
title <- c('Concrete dataset: Contribution of features', 
           'Diabetes dataset: Contribution of features')
subtitle <- c('Adding noise to feature AGE',
              'Adding noise to feature BMI')
legend <- c('Age', 'BMI')
filename <- c('noise_concrete.png', 'noise_diabetes.png')


for (i in c(1, 2)) {
  df <- data(files[i], cols[i])
  df <- df %>%
    mutate(color = ifelse(col==cols[i], 1, NA))
  plot(df, order_cols, title[i], subtitle[i], legend[i], filename[i])
}

