#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma:   zie file_name_this_file
#' Doel:        uitvoeren van een cross validation (op de train set van OT2020)
#' Auteurs:     Jules van Ligtenberg
#'              Diederik Perdok
#' Datum:       2020-02-28 (post deadline)
########################################################################################################################

rm(list=ls())

file_name_this_file <- 'VWS_Gupta_i2i_ML_doe_cross_validation_v034.R' # 

doe_cv <- function(model_source, verbose=0, debug=F, data_version=11, min_score_fold1=0, low_memory=F, save_models=F){
  cv_ver <- as.integer(substr(file_name_this_file, nchar(file_name_this_file) - 4 , nchar(file_name_this_file) - 2))

  # libraries
  library(data.table)
  library(here)
  library(mltools)
  library(stringr)
  
  root_dir           <- paste0(here(), '/')
  data_dir           <- paste0(root_dir, 'data/')
  logging_dir        <- paste0(root_dir, 'logging/')
  results_dir        <- paste0(root_dir, 'results/')
  source_dir         <- paste0(root_dir, 'src/models/')
  trained_models_dir <- paste0(root_dir, 'trained_models/')
  
  # utils
  source(paste0(root_dir, 'src/utils/VWS_Gupta_i2i_ML_utils_v002.R'))

  # settings
  freeze_and_source(paste0(root_dir, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

  file_name_train_set <- paste0('train_set_v', str_pad(data_version, 3, pad = "0") ,'.csv')
  
  if(debug){
    debug_string <- '_debug'
  }else{
    debug_string <- ''
  }
  
  cv_vector <- fread(file=paste0(data_dir, 'cv_vector.csv'), nrows=if (debug) 250000 else Inf)

  cv_vector <- cv_vector$cv_fold
  
  nfold     <- max(cv_vector)

  node_name <- Sys.info()[4] 
  user      <- Sys.info()[7] 
  
  file_name_logging <- paste0(logging_dir, 'post_deadline_',model_source, '_log_run_', format(Sys.time(), '%Y_%b_%d_%H_%M_%S'),'_on_data_v',
                              str_pad(data_version, 3, pad = "0"), '_cv_v', str_pad(cv_ver, 3, pad = "0"), 
                              '_', node_name, debug_string, '.csv')
  file_name_scores  <- paste0(results_dir, 'post_deadline_',model_source, '_scores_on_data_v', str_pad(data_version, 3, pad = "0"),
                              '_cv_v', str_pad(cv_ver, 3, pad = "0"), '_', node_name, debug_string,'.csv')

  file_name_results <- paste0(results_dir, 'post_deadline_',model_source, '_OOF_predictions_on_data_v',
                              str_pad(data_version, 3, pad = "0"), '_cv_v', str_pad(cv_ver, 3, pad = "0"), 
                              '_', node_name, debug_string, '.csv')
  
  
  file_name_model_source <- paste0(source_dir, model_source)
  if(debug){
    source(file_name_model_source)
  }else{
    freeze_and_source(file_name_model_source)
  }
  
  if(!low_memory){
    # lezen data
    if(exists('i2i_features')){
      fields <- c('target', 'gew', 'induur', 'Pseudoniem_BSN', 'zvnr', i2i_features)
      train_set <- fread(file=paste0(data_dir, file_name_train_set), select=fields, nrows=if (debug) 250000 else Inf)
    }else{
      train_set <- fread(file=paste0(data_dir, file_name_train_set), nrows=if (debug) 250000 else Inf)
    }
    if(verbose > 0) cat('Data ingelezen.\n')
  }
  
  # save alle output naar het scherm ook in een log bestand
  cur_warn_option <- getOption('warn')
  options(warn=1)
  sink() #alle voorgaande splits van de output nu stoppen
  options(warn=cur_warn_option)
  sink(file_name_logging, append=F, split=T)
  
  # Laat environment zien (na de sourcing van de progs) en bewaar in logging
  cat('sessionInfo():\n')
  print(sessionInfo())

  cat('Gestart met cross validation op', date(), '.\n')
  
  scores       <- c()
  scores_R2    <- c()
  times        <- c()
  times_train  <- c()
  times_pred   <- c()
  model_names <- c()
  for(fold in 1:nfold){
    start_time <- proc.time()[3]
    cat('Bezig met fold', fold, 'van', nfold, 'van cv van', model_source, debug_string,'\n')
    
    if(low_memory){
      # lezen data (elke fold opnieuw om totale hoeveelheid memory laag te houden)
      if(exists('i2i_features')){
        fields <- c('target', 'gew', 'induur', 'Pseudoniem_BSN', 'zvnr', i2i_features)
        train_set <- fread(file=paste0(data_dir, file_name_train_set), select=fields, nrows=if (debug) 250000 else Inf)
      }else{
        train_set <- fread(file=paste0(data_dir, file_name_train_set), nrows=if (debug) 250000 else Inf)
      }
      if(verbose > 0) cat('Data ingelezen.\n')
    }
    
    test_fold        <- train_set[cv_vector == fold, ]
    actuals          <- test_fold$target
    weights          <- test_fold$gew
    test_fold$target <- NULL
    test_fold$gew    <- NULL
    train_fold       <- train_set[cv_vector != fold, ]
    if(verbose > 0) cat('train en test fold gemaakt.\n')
    
    if(low_memory){
      # zoveel mogelijk opruimen
      rm(train_set);gc(full=T)
      if(verbose > 0) cat('train_set verwijderd en geheugen vrijgemaakt.\n')
      
      # om geheugen te besparen converteren we de BSN key naar een integer  
      train_fold$Pseudoniem_BSN <- as.integer(as.factor(train_fold$Pseudoniem_BSN))
      if(verbose > 0) cat('Pseudoniem_BSN in train_fold geconverteerd naar een integer.\n')
      
      gc(full=T)
    }

    start_time_train <- proc.time()[3]
    if(debug){
      if(grepl('_ols_|_segmented_', model_source)){
        for(col in colnames(train_set)){
          train_fold[[col]][1] <- 1 #hack om te zorgen dat OLS draait in debug mode
        }
      }
    }
    # print(sort(colnames(train_fold))) #i2i_debug
    set.seed(fold) # voor modellen met een random component
    if('verbose'  %in% names(formals('i2i_train'))){
      model  <- i2i_train(train_fold, verbose=verbose)
    }else{
      model  <- i2i_train(train_fold)
    }
    if(verbose > 0){
      cat('model getraind:\n')
      # Er zijn modellen (bv decision tree met een maxdepth > 30) die een warning genereren als je ze print.
      cur_option_warn <- getOption('warn')
      options(warn=1) 
      print(model)
      options(warn=cur_option_warn)
      cat('\n')
    }
    time_elapsed_train <- round(proc.time()[3] - start_time_train) # tijd in seconden
    
    start_time_pred <- proc.time()[3]
    cur_option_warn <- getOption('warn')
    if(grepl('_ols_|_segmented_', model_source)) options(warn=1) 
    preds  <- i2i_predict(model, test_fold)
    options(warn=cur_option_warn)
    preds <- round(preds) # we voorspellen in eurocenten
    if(verbose > 1){
      cat('de eerste 5 voorspellingen:\n')
      print(preds[1:5])
      cat('\n')
    }
    if(save_models){
      file_path_model <- paste0(trained_models_dir, 'post_deadline_', model_source,'_on_data_v', str_pad(data_version, 3, pad = "0"),
                                '_', node_name, '_fold_', fold, debug_string, '.RData')
      save(model, file = file_path_model)
      if (verbose > 0) cat(paste0('Model opgeslagen als ', file_path_model, '\n'))
      if(!debug) Sys.chmod(file_path_model, mode = "0444", use_umask = TRUE)
    }
    time_elapsed_pred <- round(proc.time()[3] - start_time_pred) # tijd in seconden
    df_predictions <- data.frame(Pseudoniem_BSN=test_fold$Pseudoniem_BSN, zvnr=test_fold$zvnr, predictions=preds)
    wrmse <- round(mltools::rmse(preds, actuals, weights), 1)
    score_R2    <- round(R2(actuals, preds, weights), 5)
    if(fold == 1){
      OOF_predictions <- df_predictions
    }else{
      OOF_predictions <- rbind(OOF_predictions, df_predictions)
    }
    time_elapsed <- round(proc.time()[3] - start_time) # tijd in seconden
    scores <- c(scores, wrmse)
    scores_R2 <- c(scores_R2, score_R2)
    times  <- c(times, time_elapsed)
    times_train  <- c(times_train, time_elapsed_train)
    times_pred  <- c(times_pred, time_elapsed_pred)
    cat('fold done in:', time_elapsed, 'seconds, weighted rmse:', wrmse, 'R2:', score_R2, '\n')
    #' Omdat sommige modellen er lang over doen om de 10-fold resultaten te berekenen, schrijven we alvast
    #' tussenresultaten weg
    df_scores <- data.frame(fold=1:fold, wrmse=scores, R2=scores_R2, time=times, train_time=times_train, 
                            pred_time=times_pred, node_name=rep(node_name, fold), user=rep(user, fold))
    write.csv(OOF_predictions, file_name_results, quote=F, row.names = F)
    write.csv(df_scores, file_name_scores, quote=F, row.names = F)
    # Als resultaten extreem tegenvallen geen resources bezet houden
    if(wrmse > 999999 && !debug){
      stop('sanity check: wrmse veel te hoog')
    }
    if(fold==1 && score_R2 < min_score_fold1){
      stop('Cross validation gestopt omdat resultaat eerste fold minder is dan het opgegeven minimum van ', min_score_fold1, '\n')
    }
  }
  
  # bevries dit bestand 
  if(!debug){
    Sys.chmod(paste0(source_dir, file_name_this_file), mode = "0444", use_umask = TRUE)
  }

  if(!debug){
    #' source file, voorspellingen en resultaten allemaal tegelijk op write protect zetten (source staat al op write
    #' protect)
    Sys.chmod(file_name_logging, mode = "0444", use_umask = TRUE)
    Sys.chmod(file_name_model_source, mode = "0444", use_umask = TRUE)
    Sys.chmod(file_name_scores, mode = "0444", use_umask = TRUE)
    Sys.chmod(file_name_results, mode = "0444", use_umask = TRUE)
  }
  # stop met het bewaren van alle output naar het scherm
  sink()
}


