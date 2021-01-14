rm(list=ls())

# libraries
library(data.table)
library(here)
library(mltools)
library(stringr)

# globals
cv_ver <- 1
data_version <- 12
this_file_version <- 8
# nieuw in 7: geen gew/gewogen bandbreedte meer en utils naar v8
# nieuw in 8: conversie voor inlezen
filename_this_file <- 
  paste0('VWS_Gupta_i2i_ML_doe_verstelbare_hyperopt_cv_v', str_pad(this_file_version, 3, pad = "0"), '.R')


doe_verstelbare_hyperopt_cv <- function(scores_hyperparams_csv_name, verbose = 1, save_models = F, debug = T){
  filenames <- get_file_names(debug)
  dirs <- get_dirs()
  
  source(paste0(dirs$root, 'src/utils/VWS_Gupta_i2i_ML_utils_v008.R'))
  freeze_and_source(paste0(dirs$root, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))
  
  # save alle output naar het scherm ook in een log bestand
  cur_warn_option <- getOption('warn')
  options(warn=1)
  sink() # alle voorgaande splits van de output nu stoppen
  options(warn=cur_warn_option)
  sink(filenames$logging, append=F, split=T)
  
  while (TRUE){
    # Lees csv, markeer volgende actieve regel en schrijf csv weer weg
    tijd_start <- proc.time()[3]
    tijd_start_string <- gsub(':', '-', gsub(' ', '-', Sys.time()))
    dt_scores_hyperparams <- lees_scores_hyperparams_csv(scores_hyperparams_csv_name)
    
    if (dt_scores_hyperparams[actief == 1, .N] > 0){
      stop("De vorige run is nog bezig of gecrasht. Pas de csv aan en zorg dat er geen actieve regel meer is.")
    }
    
    regelnr_actief <- which(dt_scores_hyperparams[, is.na(R2) & is.na(bandbreedte)])[1] # eerste zonder scores
    if (is.na(regelnr_actief)){
      cat('Klaar. Geen nieuwe regel om actief te maken.\n')
      break
    }
    
    dt_scores_hyperparams[regelnr_actief, actief := 1]
    schrijf_scores_hyperparams_csv(dt_scores_hyperparams, scores_hyperparams_csv_name)
    
    # updates op format
    if(is.null(dt_scores_hyperparams$bandbreedte_mean_folds)){
      dt_scores_hyperparams$bandbreedte_mean_folds <- NA
    }
    if(is.null(dt_scores_hyperparams$bandbreedte_sd_folds)){
      dt_scores_hyperparams$bandbreedte_sd_folds <- NA
    }
    if(is.null(dt_scores_hyperparams$bandbreedte_rond_nul)){
      dt_scores_hyperparams$bandbreedte_rond_nul <- NA
    }
    if(is.null(dt_scores_hyperparams$r2_bandbreedte_geometric.mean)){
      dt_scores_hyperparams$r2_bandbreedte_geometric.mean <- NA
    }
    if(is.null(dt_scores_hyperparams$hyperopt_versie)){
      dt_scores_hyperparams$hyperopt_versie <- NA
    }
    
    
    # Doe 10-folds cv
    hyperparam_names = setdiff(names(dt_scores_hyperparams),
                               c('source', 'R2', 'bandbreedte', 'server',
                                 'tijdsduur', 'actief', 'debug_mode','starttijd'))
    hyperparams = as.list(dt_scores_hyperparams[regelnr_actief, hyperparam_names, with=F])
    hyperparams <- hyperparams[!is.na(hyperparams)]
    cat('Start nieuwe cv met hyperparameters: ', paste(names(hyperparams), hyperparams), '\n')
    scores = doe_cv(dt_scores_hyperparams[regelnr_actief, source], hyperparams, tijd_start_string, verbose,
                    save_models, debug)
    
    # Schrijf resultaten naar csv (eerst opnieuw inlezen voor het geval het bestand door gebruiker is gewijzigd)
    dt_scores_hyperparams <- lees_scores_hyperparams_csv(scores_hyperparams_csv_name)
    if (dt_scores_hyperparams[actief == 1, .N] != 1){
      stop("Er had precies 1 run actief moeten zijn.")
    }
    regelnr_actief <- which(dt_scores_hyperparams[, actief == 1])[1]
    
    for (i in 1:length(hyperparams)){
      if(dt_scores_hyperparams[regelnr_actief, names(hyperparams)[i], with = F] != hyperparams[i]){
        stop('De actieve regel of de hyperparameters daarvan is/zijn tussentijds gewijzigd.')
      }
    }
    tijd_klaar <- proc.time()[3]
    dt_scores_hyperparams[regelnr_actief, `:=`(actief = 0,
                                               R2 = scores$R2,
                                               bandbreedte = scores$bandbreedte,
                                               bandbreedte_mean_folds = scores$bandbreedte_mean_folds,
                                               bandbreedte_sd_folds = scores$bandbreedte_sd_folds,
                                               bandbreedte_rond_nul = scores$bandbreedte_rond_nul,
                                               r2_bandbreedte_geometric.mean = 
                                                 round(((1 - scores$R2) * scores$bandbreedte)^.5),
                                               server = Sys.info()[4],
                                               tijdsduur = round(tijd_klaar - tijd_start, 0),
                                               starttijd = tijd_start_string,
                                               debug_mode = debug,
                                               hyperopt_versie = this_file_version)]
    schrijf_scores_hyperparams_csv(dt_scores_hyperparams, scores_hyperparams_csv_name)
  }
  
  # stop met het bewaren van alle output naar het scherm
  sink()
}

lees_scores_hyperparams_csv <- function(scores_hyperparams_csv_name){
  dirs <- get_dirs()
  
  # conversie voor serieus inlezen
  dt_scores_hyperparams <- fread(paste0(dirs$results, scores_hyperparams_csv_name),
                                 sep = ',',
                                 header = T,
                                 na.strings = c('NA'),
                                 stringsAsFactors = F,
                                 encoding = 'UTF-8')
  # hernoemen "gewogen" kolommen
  old <- c("gew_bandbreedte", "gew_bandbreedte_mean_folds", "gew_bandbreedte_sd_folds", "gew_bandbreedte_rond_nul",
           "r2_gew_bandbreedte_geometric", "r2_gew_bandbreedte_geometric.mean")
  new <- c("bandbreedte", "bandbreedte_mean_folds", "bandbreedte_sd_folds", "bandbreedte_rond_nul",
           "r2_bandbreedte_geometric", "r2_bandbreedte_geometric.mean")
  setnames(dt_scores_hyperparams, old, new, skip_absent = TRUE)
  fwrite(dt_scores_hyperparams,
         paste0(dirs$results, scores_hyperparams_csv_name),
         sep = ',',
         eol = '\n',
         na = 'NA'
  )
  
  dt_scores_hyperparams <- fread(paste0(dirs$results, scores_hyperparams_csv_name),
                                 sep = ',',
                                 header = T,
                                 na.strings = c('NA'),
                                 stringsAsFactors = F,
                                 colClasses = c('source' = 'character',
                                                'R2' = 'double',
                                                'bandbreedte' = 'double',
                                                'bandbreedte_mean_folds' = 'double',
                                                'bandbreedte_sd_folds' = 'double',
                                                'bandbreedte_rond_nul' = 'logical',
                                                'r2_bandbreedte_geometric.mean' = 'double',
                                                'server' = 'character',
                                                'tijdsduur' = 'double',
                                                'actief' = 'integer',
                                                'starttijd' = 'character',
                                                'debug_mode' = 'integer',
                                                'hyperopt_versie' = 'integer'),
                                 encoding = 'UTF-8')
  
  return(dt_scores_hyperparams)
}

schrijf_scores_hyperparams_csv <- function(dt_scores_hyperparams, scores_hyperparams_csv_name){
  dirs <- get_dirs()
  fwrite(dt_scores_hyperparams,
         paste0(dirs$results, scores_hyperparams_csv_name),
         sep = ',',
         eol = '\n',
         na = 'NA'
         )
}

get_dirs <- function(){
  root_dir           <- paste0(here(), '/')
  data_dir           <- paste0(root_dir, 'data/')
  logging_dir        <- paste0(root_dir, 'logging/')
  R_dir              <- paste0(root_dir, 'src/R/')
  results_dir        <- paste0(root_dir, 'results/')
  source_dir         <- paste0(root_dir, 'src/models/')
  trained_models_dir <- paste0(root_dir, 'trained_models/')
  
  return(list(root = root_dir,
              data = data_dir,
              logging = logging_dir,
              R = R_dir,
              results = results_dir,
              source = source_dir,
              trained_models = trained_models_dir))
}

get_file_names <- function(debug, model_source = 'onbekend_model', fold = 'nvt', starttijd = NULL){
  if (debug) debug_string <- '_debug'
  else debug_string <- ''
  
  node_name <- Sys.info()[4]

  dirs <- get_dirs()
  
  file_name_train_set <- paste0(dirs$data, 'train_set_v', str_pad(data_version, 3, pad = "0") ,'.csv')
  file_name_model_source <- paste0(dirs$source, model_source)
  file_name_cv_vector <- paste0(dirs$data, 'cv_vector.csv')
  
  file_name_logging <- paste0(dirs$logging, 'vervolgvraag1_hyperopt_', '_log_run_',
                              format(Sys.time(), '%Y_%b_%d_%H_%M_%S'), '_on_data_v',
                              str_pad(data_version, 3, pad = "0"), '_cv_v', str_pad(cv_ver, 3, pad = "0"),
                              '_', node_name, debug_string, '.txt')
  file_name_scores <- paste0(dirs$results, 'vervolgvraag1_', model_source, '_scores_on_data_v',
                              str_pad(data_version, 3, pad = "0"), '_cv_v', str_pad(cv_ver, 3, pad = "0"), '_',
                              node_name, debug_string, '_gestart_op_', starttijd, '.csv')
  file_name_results <- paste0(dirs$results, 'vervolgvraag1_', model_source, '_OOF_predictions_on_data_v',
                              str_pad(data_version, 3, pad = "0"), '_cv_v', str_pad(cv_ver, 3, pad = "0"), 
                              '_', node_name, debug_string, '_gestart_op_', starttijd, '.csv')
  file_path_saved_model <- paste0(dirs$trained_models, 'vervolgvraag1_', model_source,'_on_data_v',
                                  str_pad(data_version, 3, pad = "0"), '_', node_name, '_fold_', fold, debug_string,
                                  '_gestart_op_', starttijd, '.RData')
  
  return(list(train_set = file_name_train_set,
              saved_model = file_path_saved_model,
              model_source = file_name_model_source,
              cv_vector = file_name_cv_vector,
              logging = file_name_logging,
              scores = file_name_scores,
              results = file_name_results,
              this = paste0(dirs$R, filename_this_file)))
}

doe_cv <- function(model_source, hyperparams, starttijd, verbose=1, save_models = F, debug = T){
  dirs <- get_dirs()
  filenames <- get_file_names(debug, model_source, starttijd=starttijd)
  
  cv_vector <- fread(file=filenames$cv_vector, nrows=if (debug) 25000 else Inf)
  cv_vector <- cv_vector$cv_fold
  
  nfold     <- max(cv_vector)
  
  node_name <- Sys.info()[4] 
  user      <- Sys.info()[7] 
  
  if(exists('i2i_features')) rm(i2i_features, pos=sys.frame(0))
  if(debug){
    debug_string <- '_debug'
    source(filenames$model_source)
  }else{
    debug_string <- ''
    freeze_and_source(filenames$model_source)
  }
  
  # lezen data
  if(exists('i2i_features')){
    fields <- c('target', 'gew', 'induur', 'Pseudoniem_BSN', 'zvnr', 'rdragnw', i2i_features)
    train_set <- fread(file = filenames$train_set, select=fields, nrows=if (debug) 25000 else Inf)
  }else{
    train_set <- fread(file = filenames$train_set, nrows=if (debug) 25000 else Inf)
  }
  if(verbose > 0) cat('Data ingelezen.\n')
  
  # Laat environment zien (na de sourcing van de progs) en bewaar in logging
  cat('sessionInfo():\n')
  print(sessionInfo())
  
  cat('Gestart met cross validation op', date(), '.\n')
  scores                 <- c()
  scores_R2              <- c()
  scores_bandbreedte <- c()
  times                  <- c()
  times_train            <- c()
  times_pred             <- c()
  model_names            <- c()
  for(fold in 1:nfold){
    start_time <- proc.time()[3]
    cat('Bezig met fold', fold, 'van', nfold, 'van cv van', model_source, debug_string,'\n')
    filenames <- get_file_names(debug, model_source, fold, starttijd)

    test_fold         <- train_set[cv_vector == fold, ]
    actuals           <- test_fold$target
    weights           <- test_fold$gew
    verzekeraar       <- test_fold$rdragnw
    test_fold$target  <- NULL
    test_fold$gew     <- NULL
    test_fold$rdragnw <- NULL
    train_fold        <- train_set[cv_vector != fold, ]
    if(verbose > 0) cat('train en test fold gemaakt.\n')
    
    start_time_train <- proc.time()[3]
    if(debug){
      if(grepl('_ols_|_segmented_', model_source)){
        for(col in colnames(train_set)){
          train_fold[[col]][1] <- 1 # hack om te zorgen dat OLS draait in debug mode
        }
      }
    }
    set.seed(fold) # voor modellen met een random component
    if ('verbose'  %in% names(formals('i2i_train'))){
      args <- vector(mode = 'list', length = length(hyperparams) + 2)
      args[[1]] <- train_fold
      args[[2]] <- verbose
      args[-(1:2)] <- hyperparams
      names(args) <- c('train_set', 'verbose', names(hyperparams))
    } else {
      args <- vector(mode = 'list', length = length(hyperparams) + 1)
      args[[1]] <- train_fold
      args[-1] <- hyperparams
      names(args) <- c('train_set', names(hyperparams))
    }
    model <- do.call(what=i2i_train, args=args)
    
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
      save(model, file = filenames$saved_model)
      if (verbose > 0) cat(paste0('Model opgeslagen als ', filenames$saved_model, '\n'))
      if(!debug) Sys.chmod(filenames$saved_model, mode = "0444", use_umask = TRUE)
    }
    time_elapsed_pred <- round(proc.time()[3] - start_time_pred) # tijd in seconden
    df_predictions <- data.frame(Pseudoniem_BSN=test_fold$Pseudoniem_BSN, zvnr=test_fold$zvnr, fold=fold,
                                 predictions=preds)
    wrmse <- round(mltools::rmse(preds, actuals, weights), 1)
    score_R2 <- round(R2(actuals, preds, weights), 5)
    score_bandbreedte <- round(bandbreedte(actuals, preds, weights, verzekeraar, verbose=verbose), 0)
    if(fold == 1){
      OOF_predictions <- df_predictions
    }else{
      OOF_predictions <- rbind(OOF_predictions, df_predictions)
    }
    time_elapsed <- round(proc.time()[3] - start_time) # tijd in seconden
    scores <- c(scores, wrmse)
    scores_R2 <- c(scores_R2, score_R2)
    scores_bandbreedte <- c(scores_bandbreedte, score_bandbreedte)
    times <- c(times, time_elapsed)
    times_train <- c(times_train, time_elapsed_train)
    times_pred <- c(times_pred, time_elapsed_pred)
    cat('fold done in:', time_elapsed, 'seconds, weighted rmse:', wrmse, 'R2:', score_R2, '\n')
    #' Omdat sommige modellen er lang over doen om de 10-fold resultaten te berekenen, schrijven we alvast
    #' tussenresultaten weg
    df_scores <- data.frame(fold=1:fold, wrmse=scores, R2=scores_R2, bandbreedte=scores_bandbreedte, time=times,
                            train_time=times_train, pred_time=times_pred, node_name=rep(node_name, fold),
                            user=rep(user, fold))
    fwrite(OOF_predictions, filenames$results, quote=F, row.names = F)
    fwrite(df_scores, filenames$scores, quote=F, row.names = F)
  }
  
  # bereken scores OOF/totaal
  train_set_with_pred <- merge(train_set, OOF_predictions, by = c('Pseudoniem_BSN', 'zvnr')) 
  R2_OOF <- R2(train_set_with_pred$target, train_set_with_pred$predictions, train_set_with_pred$gew)
  bandbreedte_OOF <- bandbreedte(train_set_with_pred$target, train_set_with_pred$predictions,
                                 train_set_with_pred$gew, train_set_with_pred$rdragnw)
  bandbreedte_mean_folds <- mean(scores_bandbreedte)
  bandbreedte_sd_folds   <- sd(scores_bandbreedte)
  bandbreedte_rond_nul   <- bandbreedte_rond_nul(train_set_with_pred$target,
                                                 train_set_with_pred$predictions,
                                                 train_set_with_pred$gew, train_set_with_pred$rdragnw)
  # bevries dit bestand 
  if(!debug){
    Sys.chmod(filenames$this, mode = "0444", use_umask = TRUE)
  }
  
  if(!debug){
    #' source file, voorspellingen en resultaten allemaal tegelijk op write protect zetten (source staat al op write
    #' protect)
    Sys.chmod(filenames$logging, mode = "0444", use_umask = TRUE)
    Sys.chmod(filenames$model_source, mode = "0444", use_umask = TRUE)
    Sys.chmod(filenames$scores, mode = "0444", use_umask = TRUE)
    Sys.chmod(filenames$results, mode = "0444", use_umask = TRUE)
  }
  
  return(list(R2 = round(R2_OOF, 5), bandbreedte = round(bandbreedte_OOF),
              bandbreedte_mean_folds = round(bandbreedte_mean_folds), 
              bandbreedte_sd_folds = round(bandbreedte_sd_folds),
              bandbreedte_rond_nul = bandbreedte_rond_nul))
}