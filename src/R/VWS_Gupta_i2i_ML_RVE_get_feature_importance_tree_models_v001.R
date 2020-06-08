#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
# Programma: VWS_Gupta_i2i_ML_RVE_get_feature_importance_tree_models_v001.R
# Doel:      bereken relatieve feature importance
# Auteurs:   Jules van Ligtenberg
#            Diederik Perdok
########################################################################################################################

# libraries
library(data.table)
library(here)

root_dir            <- paste0(here(), '/')
logging_dir         <- paste0(root_dir, 'logging/log_train_op_gehele_data_set_geselecteerde_modellen/')
trained_models_dir  <- paste0(root_dir, 'trained_models/geselecteerde_modellen_trained/')

# utils
source(paste0(root_dir, 'src/utils/VWS_Gupta_i2i_ML_utils_v002.R'))

# settings
freeze_and_source(paste0(root_dir, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

files <- c()
files[1] <- 'VWS_Gupta_i2i_ML_decision_tree_v026.R_op_gehele_train_set_v011_vws_rve_ml_server2.RData'
files[2] <- 'VWS_Gupta_i2i_ML_RF_v002.R_op_gehele_train_set_v011_vws_rve_ml-vm.RData'
files[3] <- 'VWS_Gupta_i2i_ML_RF_v003.R_op_gehele_train_set_v011_vws_rve_ml-vm.RData'
files[4] <- 'VWS_Gupta_i2i_ML_XGB_only_OT_v001.R_op_gehele_train_set_v011_vws_rve_ml-vm.RData'
files[5] <- 'VWS_Gupta_i2i_ML_XGB_v015.R_op_gehele_train_set_v011_vws_rve_ml-vm.RData'
stopifnot(length(unique(files))==5)


for(file in files){
  cat(file, '\n')
  file_name_model <- paste0(trained_models_dir, file)
  model <- readRDS(file = file_name_model)
  feature_importance_name <- paste0(logging_dir, 'Feature_importance_', file, '.csv')
  if(grepl('_decision_tree_', file)){
    cat('decision_tree\n')
    dt_feat_imp <- data.table(Feature=names(model$variable.importance), Importance=model$variable.importance)
    dt_feat_imp <- dt_feat_imp[order(dt_feat_imp$Importance, decreasing=T),]
    print(dt_feat_imp)
    fwrite(dt_feat_imp, file=feature_importance_name, row.names=F, quote=F)
  }else if(grepl('_RF_', file)){
    cat('random forest\n')
    dt_feat_imp <- data.table(Feature=names(model$variable.importance), Importance=model$variable.importance)
    dt_feat_imp <- dt_feat_imp[order(dt_feat_imp$Importance, decreasing=T),]
    print(dt_feat_imp)
    fwrite(dt_feat_imp, file=feature_importance_name, row.names=F, quote=F)
  }else{
    cat('XGB\n')
    n <- model$n
    dt_feat_imp <- data.table(Feature=character(0), Gain=numeric(0), Cover=numeric(0), Frequency=numeric(0))
    for(i in 1:n){
      dt_new <- xgb.importance(model = model[[i]])
      dt_feat_imp <- rbind(dt_feat_imp, dt_new)
    }
    dt_feat_imp <- aggregate(.~Feature, data=dt_feat_imp, FUN=mean)
    dt_feat_imp <- dt_feat_imp[order(dt_feat_imp$Gain, decreasing=T),]
    print(dt_feat_imp)
    fwrite(dt_feat_imp, file=feature_importance_name, row.names=F, quote=F)
  }
}
