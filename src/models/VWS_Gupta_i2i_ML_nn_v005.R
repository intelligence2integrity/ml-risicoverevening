#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

#' For questions contact:

#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Bestandsnaam: VWS_Gupta_i2i_ML_nn_vxxx.R
#' Doel:         Trainen van neural netwerk op OT2020 + eventueel ruwe brondata.
#'               voorspellen van (ander) deel van de data set OT2020+ eventueel ruwe brondata.
#' Auteurs:      Jules van Ligtenberg
#'               Diederik Perdok
#' Opmerking:    - Deze file bevat alleen de code die afhankelijk is van het gekozen model. Deze file sourcen en 
#'               (gedeeltelijk) uitvoeren (via bijvoorbeeld doe_cv).
#'               - Het trainen van een model via dit script is helaas niet deterministisch. Volgens de documentatie van
#'               keras lijkt dit alleen gegarandeerd te kunnen worden als je niet parallel traint, wat praktisch geen
#'               optie is: https://keras.rstudio.com/articles/faq.html#how-can-i-obtain-reproducible-results-using-kera
#'               s-during-development. Dit zullen we nog verder onderzoeken.
########################################################################################################################

library(data.table)
library(keras)
library(parallel)

prepareer_data_features <- function(dt_x, means = NULL, stdevs = NULL){
  # Selecteer gebruikte features
  print('Selecteer features')
  nn_feats = c('lgnw', 'regsom', 'avinw', 'sesnw', 'ppanw', 'pdkg', 'sdkg', 'fkg00', 'fkg01', 'fkg02', 'fkg03',
               'fkg04', 'fkg05', 'fkg06', 'fkg07', 'fkg08', 'fkg09', 'fkg10', 'fkg11', 'fkg12', 'fkg13', 'fkg14',
               'fkg15', 'fkg16', 'fkg17', 'fkg18', 'fkg19', 'fkg20', 'fkg21', 'fkg22', 'fkg23', 'fkg24',
               'fkg25', 'fkg26', 'fkg27', 'fkg28', 'fkg29', 'fkg30', 'fkg31', 'fkg32', 'fkg33', 'fkg34',
               'fkg35', 'fkg36', 'fkg37', 'mhk', 'hkg', 'fdg', 'mvv')
  
  dt_x <- dt_x[, nn_feats, with = FALSE]
  
  # One-hot encoding (werkt ook als niet alle mogelijke waarden aanwezig zijn in de trainset)
  print('One hot encode')
  one_hot_encoding(dt_x, 'lgnw',   1:42) # functie afkomstig uit VWS_Gupta_i2i_ML_utils_vxxx.R; moet reeds gesourced
  one_hot_encoding(dt_x, 'regsom', 1:10) # zijn, bijvoorbeeld in VWS_Gupta_i2i_ML_doe_cross_validation_vxxx.R
  one_hot_encoding(dt_x, 'sesnw',  1:12)
  one_hot_encoding(dt_x, 'ppanw',  0:15)
  one_hot_encoding(dt_x, 'pdkg',   0:15)
  one_hot_encoding(dt_x, 'sdkg',   0:7)
  one_hot_encoding(dt_x, 'avinw',  0:42)
  one_hot_encoding(dt_x, 'mhk',    0:8)
  one_hot_encoding(dt_x, 'hkg',    0:10)
  one_hot_encoding(dt_x, 'fdg',    0:4)
  one_hot_encoding(dt_x, 'mvv',    0:9)
  
  # Features normaliseren. De gemiddeldes en standaarddeviaties die hiervoor gebruikt worden passen we ook toe op de
  # testset, zodat de dataprepratie identiek is in train en test. Zie i2i_train vs. i2i_predict.
  print('Converteer types')
  dt_x[, names(dt_x) := lapply(.SD, as.double)]
  
  if (is.null(means))
    print('Bereken means')
    means <- dt_x[, lapply(.SD, mean)]
  if (is.null(stdevs))
    print('Bereken stdevs')
    stdevs <- dt_x[, lapply(.SD, sd)]
  
  print('Normaliseer')
  ones <- rep(1, nrow(dt_x))
  dt_x[, names(dt_x) := (.SD - means[ones]) / (stdevs[ones] + 1e-8)] # 1e-8 bij de noemer voor numerieke stabiliteit
  
  return(list('dt_x' = dt_x, 'means' = means, 'stdevs' = stdevs))
}

definieer_en_train_model <- function(dt_train_x, dt_train_y, dt_train_w){
  # Model definieren
  print('Definieer model')
  model <- keras_model_sequential() %>%
    layer_dense(units = 256, activation = 'relu', input_shape = ncol(dt_train_x)) %>%
    layer_dense(units = 128, activation = 'relu') %>%
    layer_dense(units = 64, activation = 'relu') %>%
    layer_dense(units = 1)
  
  model %>% compile(optimizer = 'adam',
                    loss = 'mean_squared_error',  # Minimaliseren van mse is equivalent aan maximaliseren van R^2
                    weighted_metrics = list('mean_squared_error'))
  print(summary(model))
  
  # Model trainen
  print('Train model')
  
  fit(model,
      as.matrix(dt_train_x),
      dt_train_y,
      batch_size = 1024,
      epochs = 20,
      validation_split = 0,
      verbose = 1,
      shuffle = T,
      sample_weight = dt_train_w)
  
  return (model)
}

i2i_train <- function(train_set){
  # Split data.table in features, target en gewichten
  dt_train_x <- train_set[, -c('target', 'gew')]
  dt_train_y <- train_set[, target]
  dt_train_w <- train_set[, gew]
  
  # Prepareer features
  geprepareerd <- prepareer_data_features(dt_train_x)
  
  # Maak model
  model <- definieer_en_train_model(dt_train_x = geprepareerd$dt_x, dt_train_y, dt_train_w)
  attr(model, 'means') <- geprepareerd$means
  attr(model, 'stdevs') <- geprepareerd$stdevs  # Nodig om dezelfde normalisatie toe te passen op test

  return(model)
}

i2i_predict <- function(model, test_set){
  geprepareerd <- prepareer_data_features(test_set, attr(model, 'means'), attr(model, 'stdevs'))
  
  print('Voorspel')
  return(predict(model, as.matrix(geprepareerd$dt_x)))
}
