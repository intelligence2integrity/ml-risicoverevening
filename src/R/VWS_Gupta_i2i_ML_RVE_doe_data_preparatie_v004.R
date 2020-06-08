#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
# Programma: VWS_Gupta_i2i_RVE_data_preparatie_v004.R
# Doel:      data preparatie van train en test sets van OT2020
# Auteurs:   Jules van Ligtenberg
#            Diederik Perdok
########################################################################################################################

rm(list=ls())

library(data.table)

if(Sys.info()[['sysname']] == "Linux"){
  root_dir    <- '/mnt/vws_rve_ml/'
}else{
  root_dir    <- 'I:/'
}

data_dir    <- paste0(root_dir, 'data/')

file_name_train <- 'train.csv'

train_set <- fread(file=file.path(data_dir, file_name_train))

file_name_source <- paste0(root_dir, 'src/R/VWS_Gupta_i2i_ML_data_preparatie_v004.R')
Sys.chmod(file_name_source, mode = "0444", use_umask = T)
source(file_name_source)

train_set <- prepare(train_set)

file_name_train_prepared <- file.path(data_dir, 'train_set_v004.csv')
write.csv(train_set, file=file_name_train_prepared, quote=F, row.names = F)
Sys.chmod(file_name_train_prepared, mode = "0444", use_umask = T)


