########################################################################################################################
#' Programma:   zie file_name_this_file
#' Doel:        uitvoeren voorspelling (op de test set van OT2020)
#' Auteurs:     Jules van Ligtenberg
#'              Diederik Perdok
########################################################################################################################

rm(list=ls())

file_name_this_file <- 'VWS_Gupta_i2i_ML_predict_v004.R'

make_final_predictions <- function(model_source, model_name, verbose=0, debug=F){
  
  # stop als het gebruikte model niet van de opgegeven source is
  stopifnot(grepl(model_source, model_name))

  # libraries
  library(data.table)
  library(here)
  library(keras)
  library(mltools)
  library(stringr)
  
  root_dir           <- paste0(here(), '/')
  data_dir           <- paste0(root_dir, 'data/')
  logging_dir        <- paste0(root_dir, 'logging/')
  results_dir        <- paste0(root_dir, 'results/final_results_bandbreedte_minimalisatie/')
  source_dir         <- paste0(root_dir, 'src/models/')
  trained_models_dir <- paste0(root_dir, 'trained_models/')
  
  # utils
  source(paste0(root_dir, 'src/utils/VWS_Gupta_i2i_ML_utils_v008.R'))

  # settings
  freeze_and_source(paste0(root_dir, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

  if(debug){
    debug_string <- '_debug'
  }else{
    debug_string <- ''
  }
  
  node_name <- Sys.info()[4] 
  user      <- Sys.info()[7] 
  
  file_name_logging <- paste0(logging_dir, model_name, '_log_run_', format(Sys.time(), '%Y_%b_%d_%H_%M_%S'), 
                              '_on_test_data_v011_', node_name, debug_string, '.csv')
  file_name_scores  <- paste0(results_dir, model_name, '_scores_on_test_data_v012_', node_name, debug_string,'.csv')

  file_name_results <- paste0(results_dir, model_name, '_predictions_on_test_data_v012_', node_name, debug_string, '.csv')
  
  file_name_model <- paste0(trained_models_dir, model_name)
  
  file_name_model_source <- paste0(source_dir, model_source)
  if(debug){
    source(file_name_model_source)
  }else{
    freeze_and_source(file_name_model_source)
  }

  file_name_test_set <- paste0('test_set_v012.csv')
  
  # save alle output naar het scherm ook in een log bestand
  cur_warn_option <- getOption('warn')
  options(warn=1)
  sink() #alle voorgaande splits van de output nu stoppen
  options(warn=cur_warn_option)
  sink(file_name_logging, append=F, split=T)
  
  # lezen data (in debug mode slechts 100 recs om bijna zeker te weten dat we bijna niets gaan leren over de test set)
  if(exists('i2i_features')){
    fields <- c('target', 'gew', 'induur', 'Pseudoniem_BSN', 'zvnr', 'rdragnw', i2i_features)
    test_set <- fread(file=paste0(data_dir, file_name_test_set), select=fields, nrows=if (debug) 100 else Inf)
  }else{
    test_set <- fread(file=paste0(data_dir, file_name_test_set), nrows=if (debug) 100 else Inf)
  }
  if(verbose > 0) cat('Data ingelezen.\n')

  #' ophalen model
  if(grepl('_nn_bandbreedte_booster', model_source)){
    # bestaat deels uit keras neuraal netwerk, waarvoor saveRDS niet werkt. Deze is apart gesaved
    model <- readRDS(file = file_name_model)
    model$trained_base_model <- load_model_hdf5(paste0(file_name_model, '_base_nn.hdf5'))
  }else if(grepl('_nn_', model_name)[1] && !(grepl('_ols_', model_name)[1])){
    # saveRDS werkt niet voor keras neuraal netwerk
    model <- load_model_hdf5(file_name_model)
  }else{
    model <- readRDS(file = file_name_model)
  }

  # we halen target, gewicht en risicodragers uit de test set
  actuals          <- test_set$target
  weights          <- test_set$gew
  risicodragers    <- test_set$rdragnw
  test_set$target  <- NULL
  test_set$gew     <- NULL
  test_set$rdragnw <- NULL
  
  cat('Gestart met voorspellen op', date(), '.\n')
  
  start_time_pred <- proc.time()[3]

  cur_warn_option <- getOption('warn')  
  if(grepl('_ols_met_', model_source)) options(warn=1)
  preds  <- i2i_predict(model, test_set)
  options(warn=cur_warn_option)
  preds <- round(preds) # we voorspellen in eurocenten
  if(verbose > 1){
    cat('de eerste 5 voorspellingen:\n')
    print(preds[1:5])
    cat('\n')
  }
  time_elapsed_pred <- round(proc.time()[3] - start_time_pred) # tijd in seconden
  df_predictions <- data.frame(Pseudoniem_BSN=test_set$Pseudoniem_BSN, zvnr=test_set$zvnr, predictions=preds)
  wrmse <- round(mltools::rmse(preds, actuals, weights), 1)
  score_R2    <- round(R2(actuals, preds, weights), 5)
  scrore_bandbreedte <- round(bandbreedte(actuals, preds, weights, risicodragers), 0)
  cat('Voorspelling gedaan in:', time_elapsed_pred, 'seconds, weighted rmse:', wrmse, 'R2:', score_R2, 'bandbreedte:',
      scrore_bandbreedte, '\n')
  df_scores <- data.frame(wrmse=wrmse, R2=score_R2, bandbreedte=scrore_bandbreedte, pred_time=time_elapsed_pred,
                          node_name=node_name, user=user)
  fwrite(df_predictions, file_name_results, quote=F, row.names = F)
  fwrite(df_scores, file_name_scores, quote=F, row.names = F)
  # bevries dit bestand
  if(!debug){
    Sys.chmod(paste0(source_dir, file_name_this_file), mode = "0444", use_umask = TRUE)
  }

  if(!debug){
    #' voorspellingen en resultaten allemaal tegelijk op write protect zetten
    Sys.chmod(file_name_model, mode = "0444", use_umask = TRUE)         # zou al op write protect moeten staan
    Sys.chmod(file_name_model_source, mode = "0444", use_umask = TRUE)  # zou al op write protect moeten staan
    Sys.chmod(file_name_scores, mode = "0444", use_umask = TRUE)
    Sys.chmod(file_name_results, mode = "0444", use_umask = TRUE)
  }
  # stop met het bewaren van alle output naar het scherm
  sink()
  if(!debug){
    Sys.chmod(file_name_logging, mode = "0444", use_umask = TRUE)
  }
}
