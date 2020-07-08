#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

#' For questions contact:

#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
# Programma: VWS_Gupta_i2i_RVE_calc_permutation_feature_importance_vxxx.R
# Doel:      bereken van alle op de volledige trainset getrainde modellen de permutation feature importance op de test
#            set
# Auteurs:   Jules van Ligtenberg
#            Diederik Perdok
########################################################################################################################

rm(list = ls())
start_time_total <- proc.time()[3]

# libraries
library(data.table)
library(here)
library(stringr)

# dirs
root_dir           <- paste0(here(), '/')
data_dir           <- paste0(root_dir, 'data/')
logging_dir        <- paste0(root_dir, 'logging/')
trained_models_dir <- paste0(root_dir, 'trained_models/geselecteerde_modellen_trained/')
models_source_dir  <- paste0(root_dir, 'src/models/')
feat_imp_dir       <- paste0(root_dir, 'trained_models/feature_importance/')

# utils en settings
source(paste0(root_dir, 'src/utils/VWS_Gupta_i2i_ML_utils_v003.R'))
freeze_and_source(paste0(root_dir, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

versie <- 5 # versie van dit programma (wordt in het resultaatbestand gezet)

# file names
node_name <- Sys.info()[4]
feat_imp_file_names <- list.files(feat_imp_dir, pattern = '.+_perm_feat_imp_.+\\.csv')
model_file_names <- list.files(trained_models_dir, pattern = '.+_op_gehele_train_set_.+\\.RData')
model_file_names <- setdiff(model_file_names, model_file_names[grep('_debug', model_file_names)])
model_file_names <- c(grep('ML_XGB_', model_file_names, value = T),
                      grep('ML_XGB_', model_file_names, value = T, invert = T)) # XGB eerst
file_name_logging <- paste0(logging_dir, 'log_run_feature_importance_', format(Sys.time(), '%Y_%b_%d_%H_%M_%S'), 
                            '_on_test_data_v011_', node_name, '.csv')

# save alle output naar het scherm ook in een log bestand
cur_warn_option <- getOption('warn')
options(warn=1)
sink() #alle voorgaande splits van de output nu stoppen
options(warn=cur_warn_option)
sink(file_name_logging, append=F, split=T)

# berekenen en wegschrijven feature importances
for (model_file_name in model_file_names){
  start_time_model <- proc.time()[3]
  cat(paste0('Bezig met ', model_file_name, '\n'))
  model_file_path <- paste0(trained_models_dir, model_file_name)
  model_source_file_path <- paste0(models_source_dir, str_sub(model_file_name, 1, regexpr('\\.R', model_file_name) + 1))
  feat_imp_file_path <- paste0(feat_imp_dir, '_', model_file_name, '_perm_feat_imp_v',
                               str_pad(versie, width = 3, pad = '0'), '.csv')
  feat_imp_file_name_zonder_versie <- paste0(model_file_name, '_perm_feat_imp_v')
  
  if (file.exists(feat_imp_file_path)){
    cat('We slaan ', paste0(model_file_name, ' over omdat de feature importance al eerder berekend is door dit', 
                            'script\n'))
    next
  }
  
  if (any(sapply(feat_imp_file_names, function(x) grepl(feat_imp_file_name_zonder_versie, x)))){
    cat('We slaan ', paste0(model_file_name, ' over omdat de feature importance al eerder berekend is door een ', 
                            'eerdere versie van dit script\n'))
    next
  }
  
  freeze_and_source(model_source_file_path) # voor i2i_predict en i2i_features
  
  cat('Inlezen data\n') # Voor elk model opnieuw voor het geval een model (per ongeluk) by-reference de data aanpast
  test_set <- fread(paste0(data_dir, 'test_set_v011.csv'))
  
  cat('Inlezen model\n')
  # Keras neuraal netwerk is opgeslagen in ander formaat
  if (grepl('ML_nn_v', model_file_name)) model <- keras::load_model_hdf5(model_file_path)
  else model <- readRDS(model_file_path)
  
  cat('Berekenen feature importances\n')
  if(exists('i2i_features')) features <- i2i_features
  else features <- NULL
  cur_warn_option <- getOption('warn')
  options(warn = 1)
  set.seed(121) # permutaties reproduceerbaar maken
  df_feat_imp <- permutation_feature_importance(model, i2i_predict, test_set, features = features, verbose = T)
  options(warn = cur_warn_option)
  
  cat('Wegschrijven resultaat\n')
  fwrite(df_feat_imp, file = feat_imp_file_path)
  Sys.chmod(feat_imp_file_path, mode = "0444", use_umask = TRUE)
  Sys.chmod(model_file_path, mode = "0444", use_umask = TRUE)
  
  rm(i2i_train, i2i_predict)
  if(exists('i2i_features')) rm('i2i_features')
  
  time_elapsed_model <- round(proc.time()[3] - start_time_model) # tijd in seconden
  cat(paste0('Model ', model_file_name, ' klaar in ', time_elapsed_model, '\n'))
}

time_elapsed_total <- round(proc.time()[3] - start_time_total) # tijd in seconden
cat(paste0('Alle modellen klaar in ', time_elapsed_total, '\n'))

# stop met het bewaren van alle output naar het scherm
sink()
