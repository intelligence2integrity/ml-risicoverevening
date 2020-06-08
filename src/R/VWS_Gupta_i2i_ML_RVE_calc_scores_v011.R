#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
# Programma: VWS_Gupta_i2i_RVE_calc_scores_vxxx.R
# Doel:      bereken van alle OOF voorspellingen de gewogen R2
# Auteurs:   Jules van Ligtenberg
#            Diederik Perdok
########################################################################################################################

# libraries
library(data.table)
library(here)

root_dir    <- paste0(here(), '/')
data_dir    <- paste0(root_dir, 'data/')
results_dir <- paste0(root_dir, 'results/')

# utils
source(paste0(root_dir, 'src/utils/VWS_Gupta_i2i_ML_utils_v002.R'))

# settings
freeze_and_source(paste0(root_dir, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

versie <- 9 #versie van dit programma (wordt in het resultaatbestand gezet)


files <- list.files(results_dir, pattern='OOF_predictions')

train_set <- fread(file=paste0(data_dir, 'train_set_v005.csv'), select=c("Pseudoniem_BSN", "zvnr", "gew", "target"))

# conroleer of er een versie met target van bestaat
for(file in files){
  if(!grepl('debug', file)){
    debug_file <- F
    file_name_results_w_target <- paste0(results_dir, 'with_target/', sub('.csv','_with_target.csv', file))
  }else{
    debug_file <- T
    file_name_results_w_target <- paste0(results_dir, 'with_target/', sub('_debug.csv','_with_target_debug.csv', file))
  }
  if(!file.exists(file_name_results_w_target)){
    preds <- fread(file=paste0(results_dir, file))
    if(debug_file || nrow(preds) == 12036083){
      bla <- merge(preds, train_set, by = c('Pseudoniem_BSN', 'zvnr'))
      # write.csv(bla, file=file_name_results_w_target, row.names=F, quote=F)
      fwrite(bla, file=file_name_results_w_target, quote=F, row.names=F)
    }
    else if (!debug_file)
      print(paste(file_name_results_w_target, 'niet aangemaakt omdat aantal regels niet klopt. Is deze nog bezig?'))
  }
}

files <- list.files(paste0(results_dir, 'with_target/'), pattern='OOF_predictions')

file_name_total_score_per_model <- paste0(results_dir, 'total_score_per_model_v', versie, '.csv')

if(file.exists(file_name_total_score_per_model)){
  score_per_model <- fread(file_name_total_score_per_model)
}else{
  score_per_model <- data.table(model=character(0), nvoorspelling=integer(0), R2=double(0), run_time=integer(0),
                                versie=integer(0))
}

for(file in files){
  cat('Bezig met', file, '\n')
  if(!(file %in% score_per_model$model)){
    file_name_with_target <- paste0(results_dir, 'with_target/', file)
    bla <- fread(file=file_name_with_target)
    if(!grepl('debug', file_name_with_target)){
      if(nrow(bla) != 12036083) stop('Houston we have a problem.')
    }
    # nog wat meta data
    file_name_fold_info <- sub('_with_target', '', sub('OOF_predictions', 'scores', paste0(results_dir, file)))
    fold_info <- fread(file_name_fold_info)
    run_time <- sum(fold_info$time)
    score_R2    <- round(R2(bla$target, bla$predictions, bla$gew), 5)
    dt <- data.table(model=file, nvoorspelling=nrow(bla), R2=score_R2, run_time=run_time, versie=versie)
    score_per_model <- rbind(score_per_model, dt)
  }
}

setorder(score_per_model, -nvoorspelling, -R2, run_time)

write.csv(score_per_model, file=file_name_total_score_per_model, quote=F, row.names = F)

if(nrow(score_per_model) > 100){
  cat('De eerste 100 stuks:\n')
  print(score_per_model[1:100, ])
}else{
  print(score_per_model)
}




