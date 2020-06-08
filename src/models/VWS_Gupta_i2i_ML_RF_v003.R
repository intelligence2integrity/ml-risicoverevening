#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma: VWS_Gupta_i2i_ML_RF_vxxx.R
#' Doel:      gewogen trainen Random Forest op (gedeelte) van data set OT2020
#'            voorspellen van (ander) deel van data set OT2020 
#' Auteurs:   Jules van Ligtenberg
#'            Diederik Perdok
#' Opmerking: Deze file bevat alleen de code die afhankelijk is van het gekozen model. Deze file sourcen en 
#'            (gedeeltelijk) uitvoeren (via bijvoorbeeld doe_cv).
#' Opmerking: Zoals de naam al een beetje aangeeft is het proces van aanmaken van de bomen voor een groot deel random.
#'            Om een en ander te kunnen reproduceren is het nodig om voor de train een seed te zetten.
#'            
########################################################################################################################

# libraries
library(ranger)
library(Matrix)

preprocess_data <- function(dt){
  dt$geslacht <- 0
  dt$geslacht[dt$lgnw > 21] <- 1
  
  dt[is.na(leeftijd) & (lgnw == 01 | lgnw == 22), leeftijd := 0]
  dt[is.na(leeftijd) & (lgnw == 02 | lgnw == 23), leeftijd := 0.5]
  dt[is.na(leeftijd) & (lgnw == 03 | lgnw == 24), leeftijd := 2.5]
  dt[is.na(leeftijd) & (lgnw == 04 | lgnw == 25), leeftijd := 7]
  dt[is.na(leeftijd) & (lgnw == 05 | lgnw == 26), leeftijd := 12]
  dt[is.na(leeftijd) & (lgnw == 06 | lgnw == 27), leeftijd := 16]
  dt[is.na(leeftijd) & (lgnw == 07 | lgnw == 28), leeftijd := 21]
  dt[is.na(leeftijd) & (lgnw == 08 | lgnw == 29), leeftijd := 27]
  dt[is.na(leeftijd) & (lgnw == 09 | lgnw == 30), leeftijd := 32]
  dt[is.na(leeftijd) & (lgnw == 10 | lgnw == 31), leeftijd := 37]
  dt[is.na(leeftijd) & (lgnw == 11 | lgnw == 32), leeftijd := 42]
  dt[is.na(leeftijd) & (lgnw == 12 | lgnw == 33), leeftijd := 47]
  dt[is.na(leeftijd) & (lgnw == 13 | lgnw == 34), leeftijd := 52]
  dt[is.na(leeftijd) & (lgnw == 14 | lgnw == 35), leeftijd := 57]
  dt[is.na(leeftijd) & (lgnw == 15 | lgnw == 36), leeftijd := 62]
  dt[is.na(leeftijd) & (lgnw == 16 | lgnw == 37), leeftijd := 67]
  dt[is.na(leeftijd) & (lgnw == 17 | lgnw == 38), leeftijd := 72]
  dt[is.na(leeftijd) & (lgnw == 18 | lgnw == 39), leeftijd := 77]
  dt[is.na(leeftijd) & (lgnw == 19 | lgnw == 40), leeftijd := 82]
  dt[is.na(leeftijd) & (lgnw == 20 | lgnw == 41), leeftijd := 87]
  dt[is.na(leeftijd) & (lgnw == 21 | lgnw == 42), leeftijd := 92]
  
  dt[, lgnw := NULL]
  
  dt <- one_hot_encoding(dt, 'regsom', 1:10)
  dt <- one_hot_encoding(dt, 'sesnw',  1:12)
  dt <- one_hot_encoding(dt, 'ppanw',  0:15)
  dt <- one_hot_encoding(dt, 'avinw',  0:42)
  
  dt
}

#' preconditie: data.table train_set is ingelezen (dit kan de hele train set zijn of een train fold gedeelte (bij
#' een cross validation))
i2i_train <- function(train_set){
  weights <- train_set$gew
  
  # we gebruiken een vaste set van data
  train_set <- train_set[, c( 'lgnw' , 'regsom',   'avinw',   'sesnw', 'ppanw',  'pdkg',    'sdkg',  'fkg00',  'fkg01',
                              'fkg02',  'fkg03',   'fkg04',   'fkg05', 'fkg06', 'fkg07',   'fkg08',  'fkg09',  'fkg10',
                              'fkg11',  'fkg12',   'fkg13',   'fkg14', 'fkg15', 'fkg16',   'fkg17',  'fkg18',  'fkg19',
                              'fkg20',  'fkg21',   'fkg22',   'fkg23', 'fkg24', 'fkg25',   'fkg26',  'fkg27',  'fkg28',
                              'fkg29',  'fkg30',   'fkg31',   'fkg32', 'fkg33', 'fkg34',   'fkg35',  'fkg36',  'fkg37',
                              'mhk',    'hkg',     'fdg',     'mvv', 'target', 'leeftijd')]
  
  train_set <- preprocess_data(train_set)
  
  #jvldebug
  print(train_set[])
  
  # maak de train set kleiner
  train_set <- as.matrix(train_set)

  model <- ranger(dependent.variable.name='target', data=train_set, case.weights=weights, num.trees=200, 
                  min.node.size=30, mtry=20, importance='impurity')
  
  model
}

#' preconditie: data.table test_set is ingelezen (dit kan de finale test set zijn of een test fold)
i2i_predict <- function(model, test_set){
  test_set <- preprocess_data(test_set)
  
  pred <- predict(model, data=test_set)
  pred <- pred$predictions
  pred
}
