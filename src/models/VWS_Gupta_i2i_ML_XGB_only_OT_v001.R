#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma: VWS_Gupta_i2i_ML_XGB_vxxx.R
#' Doel:      gewogen trainen XGBoost op (gedeelte) van data set OT2020
#'            voorspellen van (ander) deel van data set OT2020 
#' Auteurs:   Jules van Ligtenberg
#'            Diederik Perdok
#' Opmerking: Deze file bevat alleen de code die afhankelijk is van het gekozen model. Deze file sourcen en 
#'            (gedeeltelijk) uitvoeren (via bijvoorbeeld doe_cv).
#'            
########################################################################################################################

# libraries
library(Matrix)
library(xgboost)

# het model gebruikt de volgende features
i2i_features <-  c('avinw',      'fdg',    'fkg00',    'fkg01',    'fkg02',    'fkg03',    'fkg04',    'fkg05',
                   'fkg06',    'fkg07',    'fkg08',    'fkg09',    'fkg10',    'fkg11',    'fkg12',    'fkg13',
                   'fkg14',    'fkg15',    'fkg16',    'fkg17',    'fkg18',    'fkg19',    'fkg20',    'fkg21',
                   'fkg22',    'fkg23',    'fkg24',    'fkg25',    'fkg26',    'fkg27',    'fkg28',    'fkg29',
                   'fkg30',    'fkg31',    'fkg32',    'fkg33',    'fkg34',    'fkg35',    'fkg36',    'fkg37',
                   'hkg',     'lgnw',      'mhk',      'mvv',     'pdkg',    'ppanw',   'regsom',     'sdkg',
                   'sesnw')

preprocess_data <- function(dt){
  # we gebruiken een vaste set van data
  dt <- dt[, i2i_features, with=F]

  dt$geslacht <- 0
  dt$geslacht[dt$lgnw > 21] <- 1
  
  dt[, lgnw := NULL]
  
  dt <- one_hot_encoding(dt, 'sesnw',  1:12)
  dt <- one_hot_encoding(dt, 'ppanw',  0:15)
  dt <- one_hot_encoding(dt, 'avinw',  0:42)
  
  dt
}

#' preconditie: data.table train_set is ingelezen (dit kan de hele train set zijn of een train fold gedeelte (bij
#' een cross validation))
i2i_train <- function(train_set, boost_track_length=9, colsample_bynode=20/127, colsample_bytree=0.9, eta=0.5,
                      feature_importance_name='imp', gamma=1, nsd=0, max_depth=200, min_child_weight=16,
                      subsample=0.632, totaal_induur_cap=1, verbose=1){
  max_number_of_trees <- 200

  
  # afkappen per leeftijdscategorie van alles boven het gemiddelde plus <nsd> standaard deviaties 
  if(nsd > 0){
    # we maken hier een tijdelijke leeftijdscategorie aan
    train_set$leeftijdscategorie <- train_set$lgnw
    train_set$leeftijdscategorie[train_set$leeftijdscategorie > 21] <- train_set$leeftijdscategorie[train_set$leeftijdscategorie > 21] - 21
    
    # (onderstaande allemaal in verzekerdenjaren)
    # bereken de standaarddeviatie van de target voor een leeftijdscategorie
    sd_leeftijdscategorie <- aggregate(target~leeftijdscategorie, data=train_set, FUN=sd)
    names(sd_leeftijdscategorie)[2] <- 'sd'
    if(verbose > 0) print(sd_leeftijdscategorie)
    # bereken het gemiddelde van de target voor een leeftijdscategorie
    mean_leeftijdscategorie <- aggregate(target~leeftijdscategorie, data=train_set, FUN=mean)
    names(mean_leeftijdscategorie)[2] <- 'mean'
    for(i in 1:nrow(sd_leeftijdscategorie)){
      lc <- sd_leeftijdscategorie$leeftijdscategorie[i]
      # bereken de totale kosten voor de leeftijdscategorie
      total_cost <- sum(train_set$target[train_set$leeftijdscategorie == lc])
      # bereken het antal personen in die leeftijdscategrorie
      n          <- sum(train_set$leeftijdscategorie == lc)
      lc_sd      <- sd_leeftijdscategorie$sd[i]
      lc_mean    <- mean_leeftijdscategorie$mean[i]
      # cap de target binnen de leeftijdscategorie voor alles wat meer is dan mean plus nsd keer de standaarddeviatie
      cap_upper <- lc_mean + (nsd * lc_sd)
      train_set$target[train_set$leeftijdscategorie == lc & train_set$target > cap_upper] <- cap_upper
      new_total_cost <- sum(train_set$target[train_set$leeftijdscategorie == lc])
      train_set$target[train_set$leeftijdscategorie == lc] <- train_set$target[train_set$leeftijdscategorie == lc] + (total_cost - new_total_cost) / n
      if(verbose > 0){
        cat('n', n, 'lc_mean', lc_mean, 'lc_sd', lc_sd,'\n')
        cat('total_cost    ', total_cost,'\n')
        cat('new_total_cost', new_total_cost,'\n')
      }
    }
    train_set$leeftijdscategorie <- NULL
  }
  
  # klaar maken om te kunnen comprimeren
  train_set <- train_set[, c(i2i_features, c('target', 'gew')), with=F]

  nobs <- nrow(train_set)
  
  # we comprimeren een en ander
  train_set <- train_set[, .(target = sum(target * gew) / sum(gew), gew = sum(gew)), by = i2i_features]
  
  nobs_new <- nrow(train_set)
  
  if(verbose > 0){
    cat('oorspronkelijk aantal observaties in deze train set:', nobs, '\n')
    cat('nieuw aantal observaties na samenvoegen in deze set:', nobs_new, '\n')
  }

  # weglaten feature-combinaties met een totale induur van minder of gelijk aan <totaal_induur_cap>
  
  train_set <- train_set[train_set$gew > totaal_induur_cap/365,]
  
  if(verbose > 0){
    cat('nieuw aantal na verwijderen observaties met (totale) induur', totaal_induur_cap, 'en kleiner in deze set: ', nrow(train_set), '\n')
  }
  
  if(verbose > 0) cat('start met trainen\n')
  weights <- train_set$gew
  target  <- train_set$target

  if(verbose > 0) cat('start met preprocessing data\n')
  
  train_set <- preprocess_data(train_set)

  if(verbose > 0) cat('maak speciale xgb matrix\n')
  
  dtrain <- xgb.DMatrix(data=as.matrix(train_set), label = target, weight=weights)

  if(verbose > 0) cat('verwijder oorspronkelijke train_set\n')
  
  rm(train_set)
  gc(full=T) # als er te weinig geheugen beschikbaar is kan R crashen

  
  model <- list()

  #' Er is een parameter seed maar deze wodt genegeerd in het R-package van XGBoost. In R wordt gebruik gemaakt van de 
  #' seed die gezet is door set.seed() en daardoor is het model reproduceerbaar
  params <- list(
    alpha             = 128,
    base_score        = 0,
    colsample_bynode  = colsample_bynode,
    colsample_bytree  = colsample_bytree,
    eta               = eta,
    gamma             = gamma,
    lambda            = 1,
    max_depth         = max_depth,
    min_child_weight  = min_child_weight,
    subsample         = subsample,
    tree_method       = 'exact'
  )

  n <- floor(max_number_of_trees/boost_track_length)
  if(verbose > 0) cat('We gaan', n, 'keer een boost track trainen\n')
  dt_feat_imp <- data.table(Feature=character(0), Gain=numeric(0), Cover=numeric(0), Frequency=numeric(0))
  for(i in 1:n){
    model[[i]] <- xgb.train(params=params, data=dtrain, nrounds=boost_track_length)
    if(verbose > 0) cat('boost track', i, 'getraind\n')
    dt_new <- xgb.importance(model = model[[i]])
    dt_feat_imp <- rbind(dt_feat_imp, dt_new)
  }
  if(verbose > 0){
    dt_feat_imp <- aggregate(.~Feature, data=dt_feat_imp, FUN=mean)
    dt_feat_imp <- dt_feat_imp[order(dt_feat_imp$Gain, decreasing=T),]
    print(dt_feat_imp)
    write.csv(dt_feat_imp, file=feature_importance_name, row.names=F, quote=F)
  }
  
  model$n        <- n
  
  model
}

#' preconditie: data.table test_set is ingelezen (dit kan de finale test set zijn of een test fold)
i2i_predict <- function(model, test_set, verbose=1){
  test_set <- preprocess_data(test_set)
  
  test_set <- xgb.DMatrix(as.matrix(test_set))

  preds <- list()
  for(i in 1:model$n){
    preds[[i]] <- predict(model[[i]], newdata=test_set)
  }
  preds <- as.data.frame(preds)
  names(preds)<- paste0('v', 1:model$n)
  if(verbose > 0) print(head(preds))
  pred <- rowMeans(preds)

  pred
}
