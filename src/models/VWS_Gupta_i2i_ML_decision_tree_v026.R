#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma: VWS_Gupta_i2i_ML_decision_tree_v024.R
#' Doel:      gewogen trainen decision tree op (gedeelte) van data set OT2020
#'            voorspellen van (ander) deel van data set OT2020 
#' Auteurs:   Jules van Ligtenberg
#'            Diederik Perdok
#' Opmerking: Deze file bevat alleen de code die afhankelijk is van het gekozen model. Deze file sourcen en 
#'            (gedeeltelijk) uitvoeren (via bijvoorbeeld doe_cv).
#' Historie:
#' Versie minsplit minbucket       cp preprocess data R2-OOF-score
#'     21       60         7 0.000030             nee      0.33812
#'     22       70         7 0.000030             nee      0.33684
#'     23       60         9 0.000030             nee      0.33836
#'     24       60         9 0.000015              ja  
########################################################################################################################

# libraries
library(rpart)

#' preconditie: data.table train_set is ingelezen (dit kan de hele train set zijn of een train fold gedeelte (bij
#' een cross validation))
i2i_train <- function(train_set){
  feature_list <- c( 'lgnw' , 'regsom',   'avinw',   'sesnw', 'ppanw',  'pdkg',    'sdkg',  'fkg00',  'fkg01',
                     'fkg02',  'fkg03',   'fkg04',   'fkg05', 'fkg06', 'fkg07',   'fkg08',  'fkg09',  'fkg10',
                     'fkg11',  'fkg12',   'fkg13',   'fkg14', 'fkg15', 'fkg16',   'fkg17',  'fkg18',  'fkg19',
                     'fkg20',  'fkg21',   'fkg22',   'fkg23', 'fkg24', 'fkg25',   'fkg26',  'fkg27',  'fkg28',
                     'fkg29',  'fkg30',   'fkg31',   'fkg32', 'fkg33', 'fkg34',   'fkg35',  'fkg36',  'fkg37',
                     'mhk',    'hkg',     'fdg',     'mvv')
    
  # klaar maken om te kunnen comprimeren
  train_set <- train_set[, c(feature_list, c('target', 'gew')), with=F]

  # we comprimeren een en ander
  command <- paste0('train_set[, .(target = sum(target * gew) / sum(gew), gew = sum(gew)), by = .(', paste(feature_list, collapse=","),')]')
  train_set <- eval(parse(text=command))
  
  # alternatief gevallen met een klein gewicht weglaten
  train_set <- train_set[train_set$gew > 1/365,]
  
  weights <- train_set$gew

  # we gebruiken een vaste set van data
  train_set <- train_set[, c(feature_list, c('target')), with=F]

  cur_opt_warnPartialMatchArgs <- getOption('warnPartialMatchArgs')
  options(warnPartialMatchArgs=F)
  model <- rpart(target~., data=train_set, weights=weights, method='anova', control=list(minsplit=60, 
                                                                                         minbucket=9, 
                                                                                         cp=0.000015, 
                                                                                         xval=0, 
                                                                                         maxdepth=30))
  options(warnPartialMatchArgs=cur_opt_warnPartialMatchArgs)
  
  cat('diepte berekenen:\n')
  nodes <- as.numeric(rownames(model$frame))
  cat('Maximum gerealiseerde diepte:', max(rpart:::tree.depth(nodes)), '\n')

  model
}

#' preconditie: data.table test_set is ingelezen (dit kan de finale test set zijn of een test fold)
i2i_predict <- function(model, test_set){
  
  cur_opt_warnPartialMatchDollar <- getOption('warnPartialMatchDollar')
  options(warnPartialMatchDollar=F)
  pred <- predict(model, newdata=test_set)
  options(warnPartialMatchDollar=cur_opt_warnPartialMatchDollar)
  pred
}
