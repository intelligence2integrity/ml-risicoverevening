#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

#' For questions contact:

#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
# Programma: VWS_Gupta_i2i_ML_segmented_normbedragen_vxxx.R
# Doel:      De normbedragen van alle segmenten uit het gesegmenteerde lineaire model ophalen en wegschrijven
# Auteurs:   Jules van Ligtenberg
#            Diederik Perdok
########################################################################################################################

extraheer_normbedragen_segmented <- function(saved_model_path, results_file_name, RDS_formaat = F)
{
  if (RDS_formaat) # Helaas hebben we zowel de functie 'save' als de functie 'saveRDS' gebruikt...
    model <- readRDS(saved_model_path)
  else
    load(saved_model_path)
  
  df_normb <- model[[1]]$coefficients
  for (i in 2:length(model))
    df_normb <- rbind(df_normb, model[[i]]$coefficients)
  
  df_normb <- data.frame(df_normb)
  
  min_leeft <- vector('numeric', length = length(model))
  max_leeft <- vector('numeric', length = length(model))
  for (i in 1:length(model)){
    min_leeft[i] <- attr(model, 'secties')[[i]][1]
    max_leeft[i] <- attr(model, 'secties')[[i]][2]
  }
  
  df_normb$leeftijd_min <- min_leeft
  df_normb$leeftijd_max <- max_leeft
  
  write.csv(df_normb, results_file_name, row.names = F)
}

