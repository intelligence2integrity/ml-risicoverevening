#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

#' For questions contact:

#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma: VWS_Gupta_i2i_ML_RF_v002.R
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
  #' het aantal mogelijke waarden is hier hard-coded omdat je niet wilt dat er bij een kleine train set waarden gemist
  #' worden
  
  dt$geslacht <- 0
  dt$geslacht[dt$lgnw > 21] <- 1
  
  # haal een leeftijdscategorie uit lgnw
  dt$leeftijdscategorie <- dt$lgnw
  dt$leeftijdscategorie[dt$leeftijdscategorie > 21] <- dt$leeftijdscategorie[dt$leeftijdscategorie > 21] - 21
  
  dt <- one_hot_encoding(dt, 'lgnw',   1:42)
  dt <- one_hot_encoding(dt, 'regsom', 1:10)
  dt <- one_hot_encoding(dt, 'sesnw',  1:12)
  dt <- one_hot_encoding(dt, 'ppanw',  0:15)
  dt <- one_hot_encoding(dt, 'pdkg',   0:15)
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
                              'mhk',    'hkg',     'fdg',     'mvv', 'target')]
  
  train_set <- preprocess_data(train_set)
  
  # maak de train set kleiner
  train_set <- as.matrix(train_set)

  model <- ranger(dependent.variable.name='target', data=train_set, case.weights=weights, num.trees=200, 
                  min.node.size=16, mtry=20, importance='impurity')
  
  model
}

#' preconditie: data.table test_set is ingelezen (dit kan de finale test set zijn of een test fold)
i2i_predict <- function(model, test_set){
  
  test_set <- preprocess_data(test_set)
  
  pred <- predict(model, data=test_set)
  pred <- pred$predictions
  pred
}
