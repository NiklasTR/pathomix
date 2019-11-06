load_cosmic <- function()
# loading downloaded data from COSMIC
cosmic_wes <- read_delim("data/cosmic/CosmicGenomeScreensMutantExport.tsv", 
           "\t", escape_double = FALSE, trim_ws = TRUE) %>% janitor::clean_names()

# preprocessing
cosmic_wes <- cosmic_wes 

