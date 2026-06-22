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


data <- function(file, repeated_col) {
  data <- readr::read_csv(file) %>% 
    tidyr::pivot_longer(!col) %>% 
    mutate(data = ifelse(col == repeated_col, "original", NA)) %>% 
    bind_rows(
      tibble(
        name = unique(.$name),
        col = repeated_col,
        value = .[.$col == 'correlated', 'value']$value,
        data = "correlated")) %>% 
    filter(col != 'correlated') %>% 
    tidyr::replace_na(list(data = 'other'))
  return(data)
}

plot <- function(data, title, subtitle, legend, filename) {
  data %>% 
    filter_if(is.numeric, all_vars((.) != 0)) %>% 
    ggplot(aes(x = as.factor(col), 
               fill = factor(data, levels = c('original', 'correlated', 'other')))) +
    geom_bar(aes(y = value), stat = 'identity') +
    facet_wrap(~ name) +
    labs(title = title,
         subtitle = subtitle) +
    scale_fill_manual(values = c("#04A0A7", "#964deb", "#C3C3C3")) +
    ylab("Contribution") +
    xlab("Feature") +
    guides(fill = guide_legend(title = legend)) +
    theme(legend.position = "bottom",
          axis.text.x = element_text(angle = 45, hjust = 1)) 
  
  ggsave(paste0('data/figures/', filename), device = "png", dpi = 320, 
         width = 40, height = 30, units = "cm")
  
}


files <- list.files(path = "data/output", pattern = 'correlation', full.names = TRUE)
cols <- c('age', 'bmi')
title <- c('Concrete data: Contribution of features', 
           'Diabetes data: Contribution of features')
subtitle <- c('Behaviour of feature AGE and the new feature correlated to it',
              'Behaviour of feature BMI and the new feature correlated to it')
legend <- c('Age', 'BMI')
filename <- c('correlation_concrete.png', 'correlation_diabetes.png')


for (i in c(1, 2)) {
  df <- data(files[i], cols[i])
  plot(df, title[i], subtitle[i], legend[i], filename[i])
}

