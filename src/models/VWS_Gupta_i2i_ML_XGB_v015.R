#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

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

# het model is ontwikkeld voor de volgende features
i2i_features <- c('aantal_fkg_01'  , 'aantal_fkg_02'  , 'aantal_fkg_03'  , 'aantal_fkg_04'  , 'aantal_fkg_05'  ,
                  'aantal_fkg_06'  , 'aantal_fkg_07'  , 'aantal_fkg_08'  , 'aantal_fkg_09'  , 'aantal_fkg_10'  ,
                  'aantal_fkg_11'  , 'aantal_fkg_12'  , 'aantal_fkg_13'  , 'aantal_fkg_14'  , 'aantal_fkg_16'  ,
                  'aantal_fkg_17'  , 'aantal_fkg_18'  , 'aantal_fkg_19'  , 'aantal_fkg_20'  , 'aantal_fkg_21'  ,
                  'aantal_fkg_22'  , 'aantal_fkg_23'  , 'aantal_fkg_24'  , 'aantal_fkg_25'  , 'aantal_fkg_26'  ,
                  'aantal_fkg_27'  , 'aantal_fkg_28'  , 'aantal_fkg_29'  , 'aantal_fkg_30'  , 'aantal_fkg_31'  ,
                  'aantal_fkg_32'  , 'aantal_fkg_33'  , 'aantal_fkg_34'  , 'aantal_fkg_35'  , 'aantal_fkg_36'  ,
                  'aantal_fkg_37'  , 'avinw'          , 'dx_groep_101018', 'dx_groep_101072', 'dx_groep_101150',
                  'dx_groep_101171', 'dx_groep_102019', 'dx_groep_111048', 'dx_groep_111049', 'dx_groep_111050',
                  'dx_groep_112007', 'dx_groep_113171', 'dx_groep_113172', 'dx_groep_151007', 'dx_groep_151016',
                  'dx_groep_16215' , 'dx_groep_1731'  , 'dx_groep_1732'  , 'dx_groep_1733'  , 'dx_groep_1741'  ,
                  'dx_groep_1742'  , 'dx_groep_1743'  , 'dx_groep_175'   , 'dx_groep_1750'  , 'dx_groep_176'   ,
                  'dx_groep_177'   , 'dx_groep_178'   , 'dx_groep_179'   , 'dx_groep_1801'  , 'dx_groep_1802'  ,
                  'dx_groep_21003' , 'dx_groep_21007' , 'dx_groep_21008' , 'dx_groep_21009' , 'dx_groep_21010' ,
                  'dx_groep_21011' , 'dx_groep_21013' , 'dx_groep_21014' , 'dx_groep_21015' , 'dx_groep_21016' ,
                  'dx_groep_21017' , 'dx_groep_21018' , 'dx_groep_21019' , 'dx_groep_21020' , 'dx_groep_21021' ,
                  'dx_groep_21022' , 'dx_groep_21027' , 'dx_groep_21032' , 'dx_groep_21033' , 'dx_groep_21034' ,
                  'dx_groep_21035' , 'dx_groep_21036' , 'dx_groep_21037' , 'dx_groep_21041' , 'dx_groep_21043' ,
                  'dx_groep_21048' , 'dx_groep_21049' , 'dx_groep_21055' , 'dx_groep_21056' , 'dx_groep_21057' ,
                  'dx_groep_21077' , 'dx_groep_21081' , 'dx_groep_21084' , 'dx_groep_21086' , 'dx_groep_21089' , 
                  'dx_groep_21093' , 'dx_groep_21096' , 'dx_groep_21097' , 'dx_groep_21098' , 'dx_groep_21099' ,
                  'dx_groep_211007', 'dx_groep_211048', 'dx_groep_21105' , 'dx_groep_211070', 'dx_groep_211072',
                  'dx_groep_211073', 'dx_groep_211093', 'dx_groep_211144', 'dx_groep_21115' , 'dx_groep_211171',
                  'dx_groep_222134', 'dx_groep_231008', 'dx_groep_23171' , 'dx_groep_232007', 'dx_groep_233171', 
                  'dx_groep_233371', 'dx_groep_243021', 'dx_groep_31077' , 'dx_groep_31080' , 'dx_groep_31081' ,
                  'dx_groep_31084' , 'dx_groep_31085' , 'dx_groep_31086' , 'dx_groep_31087' , 'dx_groep_31089' ,
                  'dx_groep_31097' , 'dx_groep_31171' , 'dx_groep_331070', 'dx_groep_331089', 'dx_groep_331093',
                  'dx_groep_333025', 'dx_groep_400001', 'dx_groep_400002', 'dx_groep_400003', 'dx_groep_400004',
                  'dx_groep_400006', 'dx_groep_400007', 'dx_groep_400008', 'dx_groep_41007' , 'dx_groep_41013' ,
                  'dx_groep_41048' , 'dx_groep_41097' , 'dx_groep_41105' , 'dx_groep_41110' , 'dx_groep_41111' ,
                  'dx_groep_42089' , 'dx_groep_500001', 'dx_groep_500002', 'dx_groep_500003', 'dx_groep_500004',
                  'dx_groep_500005', 'dx_groep_500006', 'dx_groep_500007', 'dx_groep_500008', 'dx_groep_500009',
                  'dx_groep_500010', 'dx_groep_500011', 'dx_groep_500012', 'dx_groep_500013', 'dx_groep_500014',
                  'dx_groep_500015', 'dx_groep_500016', 'dx_groep_500019', 'dx_groep_500020', 'dx_groep_500021',
                  'dx_groep_500022', 'dx_groep_500023', 'dx_groep_500024', 'dx_groep_500025', 'dx_groep_500026',
                  'dx_groep_500027', 'dx_groep_500028', 'dx_groep_500029', 'dx_groep_500030', 'dx_groep_500031',
                  'dx_groep_500032', 'dx_groep_500033', 'dx_groep_500034', 'dx_groep_500035', 'dx_groep_500036',
                  'dx_groep_51048' , 'dx_groep_51049' , 'dx_groep_51050' , 'dx_groep_53098' , 'dx_groep_600010',
                  'dx_groep_600020', 'dx_groep_600021', 'dx_groep_600030', 'dx_groep_600040', 'dx_groep_600050',
                  'dx_groep_71009' , 'dx_groep_71010' , 'dx_groep_71011' , 'dx_groep_71025' , 'dx_groep_71032' ,
                  'dx_groep_71034' , 'dx_groep_71035' , 'dx_groep_71036' , 'dx_groep_71037' , 'dx_groep_71041' ,
                  'dx_groep_71098' , 'dx_groep_72007' , 'dx_groep_83070' , 'dx_groep_91009' , 'dx_groep_91010' ,
                  'dx_groep_91011' , 'dx_groep_91013' , 'dx_groep_91014' , 'dx_groep_91015' , 'dx_groep_91021' ,
                  'dx_groep_91027' , 'dx_groep_91032' , 'dx_groep_91036' , 'dx_groep_91037' , 'dx_groep_91041' ,
                  'dx_groep_91043' , 'dx_groep_91049' , 'dx_groep_91070' , 'dx_groep_91092' , 'dx_groep_91095' ,
                  'dx_groep_91096' , 'dx_groep_91097' , 'dx_groep_91134' , 'dx_groep_91150' , 'dx_groep_91151' ,
                  'dx_groep_92007' , 'dx_groep_92153' , 'dx_groep_93049' , 'dx_groep_93050' , 'dx_groep_93051' ,
                  'fdg'            , 'fkg00'          , 'fkg01'          , 'fkg02'          , 'fkg03'          , 
                  'fkg04'          , 'fkg05'          , 'fkg06'          , 'fkg07'          , 'fkg08'          ,
                  'fkg09'          , 'fkg10'          , 'fkg11'          , 'fkg12'          , 'fkg13'          ,
                  'fkg14'          , 'fkg15'          , 'fkg16'          , 'fkg17'          , 'fkg18'          ,
                  'fkg19'          , 'fkg20'          , 'fkg21'          , 'fkg22'          , 'fkg23'          ,
                  'fkg24'          , 'fkg25'          , 'fkg26'          , 'fkg27'          , 'fkg28'          ,
                  'fkg29'          , 'fkg30'          , 'fkg31'          , 'fkg32'          , 'fkg33'          ,
                  'fkg34'          , 'fkg35'          , 'fkg36'          , 'fkg37'          , 'hkg'            ,
                  'leeftijd2'      , 'lgnw'           , 'mhk'            , 'mvv'            , 'n_operatief'    ,
                  'n_verpleeg'     , 'pdkg'           , 'ppanw'          , 'regsom'         , 'sdkg'           ,
                  'sesnw')

preprocess_data <- function(dt){
  # we gebruiken een vaste set van data
  dt <- dt[, i2i_features, with=F]

  dt$geslacht <- 0
  dt$geslacht[dt$lgnw > 21] <- 1
  
  dt[, lgnw := NULL]
  
  # dt <- one_hot_encoding(dt, 'regsom', 1:50)
  dt <- one_hot_encoding(dt, 'sesnw',  1:12)
  dt <- one_hot_encoding(dt, 'ppanw',  0:15)
  dt <- one_hot_encoding(dt, 'avinw',  0:42)
  
  dt
}

#' preconditie: data.table train_set is ingelezen (dit kan de hele train set zijn of een train fold gedeelte (bij
#' een cross validation))
i2i_train <- function(train_set, boost_track_length=9, colsample_bytree=0.9, eta=0.4, feature_importance_name='imp', 
                      gamma=1, nsd=0, max_min_child_weight=32, subsample=0.632, totaal_induur_cap=1, verbose=1){
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
  avg     <- weighted.mean(target)

  if(verbose > 0) cat('start met preprocessing data\n')
  
  train_set <- preprocess_data(train_set)

  if(verbose > 0) cat('maak speciale xgb matrix\n')
  
  dtrain <- xgb.DMatrix(data=as.matrix(train_set), label = target, weight=weights)

  if(verbose > 0) cat('verwijder oorspronkelijke train_set\n')
  
  rm(train_set)
  gc(full=T) # als er te weinig geheugen beschikbaar is kan R crashen

  
  model <- list()
  

  n <- floor(max_number_of_trees/boost_track_length)
  if(verbose > 0) cat('We gaan', n, 'keer een boost track trainen\n')
  dt_feat_imp <- data.table(Feature=character(0), Gain=numeric(0), Cover=numeric(0), Frequency=numeric(0))
  for(i in 1:n){
    #' Er is een parameter seed maar deze wodt genegeerd in het R-package van XGBoost. In R wordt gebruik gemaakt van de 
    #' seed die gezet is door set.seed() en daardoor is het model reproduceerbaar
    params <- list(
      alpha             = 128,
      base_score        = 0,
      # base_score        = median_target,
      colsample_bynode  = 20/127,
      colsample_bytree  = colsample_bytree,
      eta               = eta,
      gamma             = gamma,
      lambda            = 1,
      max_depth         = 10000,
      min_child_weight  = max_min_child_weight * i/n,
      subsample         = subsample,
      tree_method       = 'exact'
    )
    model[[i]] <- xgb.train(params=params, data=dtrain, nrounds=boost_track_length)
    if(verbose > 0) cat('boost track', i, 'getraind\n')
    dt_new <- xgb.importance(model = model[[i]])
    dt_feat_imp <- rbind(dt_feat_imp, dt_new)
  }
  dt_feat_imp <- aggregate(.~Feature, data=dt_feat_imp, FUN=mean)
  dt_feat_imp <- dt_feat_imp[order(dt_feat_imp$Gain, decreasing=T),]
  if(verbose > 0) print(dt_feat_imp)
  write.csv(dt_feat_imp, file=feature_importance_name, row.names=F, quote=F)
  
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
