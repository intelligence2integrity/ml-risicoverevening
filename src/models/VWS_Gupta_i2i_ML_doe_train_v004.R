########################################################################################################################
#' Programma:   zie file_name_this_file
#' Doel:        model trainen op volledige trainset
#' Auteurs:     Jules van Ligtenberg
#'              Diederik Perdok
########################################################################################################################

rm(list=ls())

file_name_this_file <- 'VWS_Gupta_i2i_ML_doe_train_v004.R'

doe_train <- function(model_source, verbose=0, debug=F, data_version=12, save_model=T){
  # libraries
  library(data.table)
  library(mltools)
  library(stringr)
  library(here)
  
  root_dir           <- paste0(here(), '/')
  data_dir           <- paste0(root_dir, 'data/')
  logging_dir        <- paste0(root_dir, 'logging/')
  source_dir         <- paste0(root_dir, 'src/models/')
  trained_models_dir <- paste0(root_dir, 'trained_models/')
  
  # utils
  source(paste0(root_dir, 'src/utils/VWS_Gupta_i2i_ML_utils_v008.R'))

  # settings
  freeze_and_source(paste0(root_dir, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

  file_name_train_set <- paste0('train_set_v', str_pad(data_version, 3, pad = "0") ,'.csv')
  
  if(debug){
    debug_string <- '_debug'
  }else{
    debug_string <- ''
  }
  
  node_name <- Sys.info()[4]
  
  characteristics <-  paste0(format(Sys.time(), '%Y_%b_%d_%H_%M_%S'))

  file_name_logging <- paste0(logging_dir, model_source, '_log_run_', characteristics,'_on_data_v',
                              str_pad(data_version, 3, pad = "0"), '_gehele_train_set_', node_name, debug_string, '.csv')
  file_name_model <- paste0(trained_models_dir, model_source, '_op_gehele_train_set_v', str_pad(data_version, 3, pad = "0"),
                            '_', node_name, debug_string, '.RData')
  feature_importance_name <- paste0(logging_dir, model_source, characteristics, '_feature_importance', debug_string, '.csv')
  
  
  file_name_model_source <- paste0(source_dir, model_source)
  if(debug){
    source(file_name_model_source)
  }else{
    freeze_and_source(file_name_model_source)
  }
  
  # lezen data afhankelijk van het bestaan van i2i_features in de huidige environment
  if(exists('i2i_features')){
    fields <- c('target', 'gew', 'induur', 'Pseudoniem_BSN', 'zvnr', i2i_features)
    train_set <- fread(file=paste0(data_dir, file_name_train_set), select=fields, nrows=if (debug) 25000 else Inf)
  }
  else
    train_set <- fread(file=paste0(data_dir, file_name_train_set), nrows=if (debug) 25000 else Inf)
  
  # save alle output naar het scherm ook in een log bestand
  cur_warn_option <- getOption('warn')
  options(warn=1)
  sink() #alle voorgaande splits van de output nu stoppen
  options(warn=cur_warn_option)
  sink(file_name_logging, append=F, split=T)
  
  # Laat environment zien (na de sourcing van de progs) en bewaar in logging
  cat('sessionInfo():\n')
  print(sessionInfo())
  
  cat('Gestart met train', date(), '.\n')
  start_time_train <- proc.time()[3]
  set.seed(2020)
  # de gebruikte modellen hebben of zowel verbose als feature_importance_name of geen van beiden:
  if('verbose'  %in% names(formals('i2i_train')) && 'feature_importance_name'  %in% names(formals('i2i_train'))){
    model  <- i2i_train(train_set, verbose=verbose, feature_importance_name=feature_importance_name)
  }else{
    model  <- i2i_train(train_set)
  }
  if(verbose > 0){
    cat('model getraind:\n')
    print(model)
    cat('\n')
  }

  if(save_model){
    if ('keras.engine.training.Model' %in% class(model)){
      # saveRDS werkt niet voor keras neuraal netwerk
      save_model_hdf5(model, file_name_model)
    }else if (class(model) == 'lm'){ # is ols
      # Uitslopen van zaken die niet nodig zijn voor voorspellen en de filesize heel groot maken,
      # zoals de complete trainingsset
      attr(model$terms, ".Environment") <- NULL
      model$qr$qr <- NULL
      model$model <- NULL
      saveRDS(model, file = file_name_model)
    }else if (grepl('_nn_bandbreedte_booster', model_source)){
      # bestaat deels uit keras neuraal netwerk, waarvoor saveRDS niet werkt. Deze saven we apart
      save_model_hdf5(model$trained_base_model, paste0(file_name_model, '_base_nn.hdf5'))
      model$trained_base_model <- NULL
      saveRDS(model, file = file_name_model)
    }else {
     # Geen speciale stappen vereist voor opslaan
     saveRDS(model, file = file_name_model)
    }
    if (verbose > 0) cat(paste0('Model opgeslagen als ', file_name_model, '\n'))
  }
  time_elapsed <- round(proc.time()[3] - start_time_train) # tijd in seconden
  cat('Done in:', time_elapsed, 'seconds\n')
  
  # bevries dit bestand 
  if(!debug){
    Sys.chmod(paste0(source_dir, file_name_this_file), mode = "0444", use_umask = TRUE)
  }

  if(!debug){
    #' source file, voorspellingen en resultaten allemaal tegelijk op write protect zetten (source staat al op write
    #' protect)
    Sys.chmod(file_name_logging, mode = "0444", use_umask = TRUE)
    Sys.chmod(file_name_model_source, mode = "0444", use_umask = TRUE)
    Sys.chmod(file_name_model, mode = "0444", use_umask = TRUE)
  }
  
  # stop met het bewaren van alle output naar het scherm
  sink()
}
