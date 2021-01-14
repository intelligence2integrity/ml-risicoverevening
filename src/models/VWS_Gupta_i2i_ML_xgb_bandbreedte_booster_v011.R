########################################################################################################################
#' Bestandsnaam: VWS_Gupta_i2i_ML_bandbreedte_booster_vxxx.R
#' Doel:         
#' Auteurs:      Jules van Ligtenberg
#'               Diederik Perdok
#' versie v010 t.o.v. v009: aanpassingen voor probleem: Cholmod error 'problem too large
#' versie v011 t.o.v. v010: nog meer aanpassingen voor probleem: Cholmod error 'problem too large
########################################################################################################################

library(data.table)
library(glmnet)
library(here)

# het model is ontwikkeld voor de volgende features (het model waarvan de voorspellingen gecorrigeerd worden
# mag zelf wel andere features gebruiken)
boost_model_features <- c('avinw',           'dx_groep_101018', 'dx_groep_101072', 'dx_groep_101150',
                          'dx_groep_101171', 'dx_groep_102019', 'dx_groep_111048', 'dx_groep_111049',
                          'dx_groep_111050', 'dx_groep_112007', 'dx_groep_113171', 'dx_groep_113172', 
                          'dx_groep_151007', 'dx_groep_151016', 'dx_groep_16215' , 'dx_groep_1731'  ,
                          'dx_groep_1732'  , 'dx_groep_1733'  , 'dx_groep_1741'  , 'dx_groep_1742'  ,
                          'dx_groep_1743'  , 'dx_groep_175'   , 'dx_groep_1750'  , 'dx_groep_176'   ,
                          'dx_groep_177'   , 'dx_groep_178'   , 'dx_groep_179'   , 'dx_groep_1801'  ,
                          'dx_groep_1802'  , 'dx_groep_21003' , 'dx_groep_21007' , 'dx_groep_21008' ,
                          'dx_groep_21009' , 'dx_groep_21010' , 'dx_groep_21011' , 'dx_groep_21013' ,
                          'dx_groep_21014' , 'dx_groep_21015' , 'dx_groep_21016' , 'dx_groep_21017' ,
                          'dx_groep_21018' , 'dx_groep_21019' , 'dx_groep_21020' , 'dx_groep_21021' ,
                          'dx_groep_21022' , 'dx_groep_21027' , 'dx_groep_21032' , 'dx_groep_21033' ,
                          'dx_groep_21034' , 'dx_groep_21035' , 'dx_groep_21036' , 'dx_groep_21037' ,
                          'dx_groep_21041' , 'dx_groep_21043' , 'dx_groep_21048' , 'dx_groep_21049' ,
                          'dx_groep_21055' , 'dx_groep_21056' , 'dx_groep_21057' , 'dx_groep_21077' ,
                          'dx_groep_21081' , 'dx_groep_21084' , 'dx_groep_21086' , 'dx_groep_21089' , 
                          'dx_groep_21093' , 'dx_groep_21096' , 'dx_groep_21097' , 'dx_groep_21098' ,
                          'dx_groep_21099' , 'dx_groep_211007', 'dx_groep_211048', 'dx_groep_21105' ,
                          'dx_groep_211070', 'dx_groep_211072', 'dx_groep_211073', 'dx_groep_211093',
                          'dx_groep_211144', 'dx_groep_21115' , 'dx_groep_211171', 'dx_groep_222134',
                          'dx_groep_231008', 'dx_groep_23171' , 'dx_groep_232007', 'dx_groep_233171', 
                          'dx_groep_233371', 'dx_groep_243021', 'dx_groep_31077' , 'dx_groep_31080' , 
                          'dx_groep_31081' , 'dx_groep_31084' , 'dx_groep_31085' , 'dx_groep_31086' ,
                          'dx_groep_31087' , 'dx_groep_31089' , 'dx_groep_31097' , 'dx_groep_31171' ,
                          'dx_groep_331070', 'dx_groep_331089', 'dx_groep_331093', 'dx_groep_333025', 
                          'dx_groep_400001', 'dx_groep_400002', 'dx_groep_400003', 'dx_groep_400004',
                          'dx_groep_400006', 'dx_groep_400007', 'dx_groep_400008', 'dx_groep_41007' ,
                          'dx_groep_41013' , 'dx_groep_41048' , 'dx_groep_41097' , 'dx_groep_41105' ,
                          'dx_groep_41110' , 'dx_groep_41111' , 'dx_groep_42089' , 'dx_groep_500001',
                          'dx_groep_500002', 'dx_groep_500003', 'dx_groep_500004', 'dx_groep_500005',
                          'dx_groep_500006', 'dx_groep_500007', 'dx_groep_500008', 'dx_groep_500009',
                          'dx_groep_500010', 'dx_groep_500011', 'dx_groep_500012', 'dx_groep_500013',
                          'dx_groep_500014', 'dx_groep_500015', 'dx_groep_500016', 'dx_groep_500019',
                          'dx_groep_500020', 'dx_groep_500021', 'dx_groep_500022', 'dx_groep_500023',
                          'dx_groep_500024', 'dx_groep_500025', 'dx_groep_500026', 'dx_groep_500027',
                          'dx_groep_500028', 'dx_groep_500029', 'dx_groep_500030', 'dx_groep_500031',
                          'dx_groep_500032', 'dx_groep_500033', 'dx_groep_500034', 'dx_groep_500035',
                          'dx_groep_500036', 'dx_groep_51048' , 'dx_groep_51049' , 'dx_groep_51050' ,
                          'dx_groep_53098' , 'dx_groep_600010', 'dx_groep_600020', 'dx_groep_600021',
                          'dx_groep_600030', 'dx_groep_600040', 'dx_groep_600050', 'dx_groep_71009' ,
                          'dx_groep_71010' , 'dx_groep_71011' , 'dx_groep_71025' , 'dx_groep_71032' ,
                          'dx_groep_71034' , 'dx_groep_71035' , 'dx_groep_71036' , 'dx_groep_71037' ,
                          'dx_groep_71041' , 'dx_groep_71098' , 'dx_groep_72007' , 'dx_groep_83070' ,
                          'dx_groep_91009' , 'dx_groep_91010' , 'dx_groep_91011' , 'dx_groep_91013' ,
                          'dx_groep_91014' , 'dx_groep_91015' , 'dx_groep_91021' , 'dx_groep_91027' ,
                          'dx_groep_91032' , 'dx_groep_91036' , 'dx_groep_91037' , 'dx_groep_91041' ,
                          'dx_groep_91043' , 'dx_groep_91049' , 'dx_groep_91070' , 'dx_groep_91092' ,
                          'dx_groep_91095' , 'dx_groep_91096' , 'dx_groep_91097' , 'dx_groep_91134' , 
                          'dx_groep_91150' , 'dx_groep_91151' , 'dx_groep_92007' , 'dx_groep_92153' ,
                          'dx_groep_93049' , 'dx_groep_93050' , 'dx_groep_93051' , 'fdg',
                          'fkg00',           'fkg01',           'fkg02',           'fkg03', 
                          'fkg04',           'fkg05',           'fkg06',           'fkg07', 
                          'fkg08',           'fkg09',           'fkg10',           'fkg11',
                          'fkg12',           'fkg13',           'fkg14',           'fkg15',
                          'fkg16',           'fkg17',           'fkg18',           'fkg19', 
                          'fkg20',           'fkg21',           'fkg22',           'fkg23',
                          'fkg24',           'fkg25',           'fkg26',           'fkg27',
                          'fkg28',           'fkg29',           'fkg30',           'fkg31',
                          'fkg32',           'fkg33',           'fkg34',           'fkg35',
                          'fkg36',           'fkg37',           'hkg',             'lgnw',
                          'mhk',             'mvv',             'pdkg',            'ppanw',
                          'regsom',          'sdkg',            'sesnw')

one_hot_encode_features <- function(dt_x){
  # One-hot encoding (werkt ook als niet alle mogelijke waarden aanwezig zijn in de trainset)
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
  
  return(dt_x)
}

prepareer_data_train_features <- function(dt_x, dt_w, dt_vz, rdragnw_to_keep){
  dt_x <- one_hot_encode_features(dt_x)
  
  # Maak geaggregeerde features op verzekeraarniveau
  dt_agg_x <- cbind(dt_vz, dt_x, dt_w)
  dt_agg_x <- dt_agg_x[rdragnw %in% rdragnw_to_keep]
  dt_agg_x <- dt_agg_x[, lapply(.SD, function(x) sum(x * gew)), by=rdragnw]
  dt_agg_x <- dt_agg_x[, -c('rdragnw', 'gew')]
  return(dt_agg_x)
}

prepareer_data_train_target <- function(dt_y, dt_w, dt_vz, basis_model_preds, boost_threshold){
  # Gewichten en eerdere predicties toevoegen aan dt_y
  dt_y <- cbind(dt_y, dt_w)
  dt_y[, predictions := basis_model_preds] # basis_model_preds staan al in de goede volgorde
  
  # Integers zijn slechts 32 bits lang in R. Converteer eurocenten dus naar floats.
  dt_y <- dt_y[, .(target, gew, predictions)][, `:=`(target = as.numeric(target),
                                                     predictions = as.numeric(predictions),
                                                     gew = as.numeric(gew))]
  # Maak geaggregeerde target op verzekeraarniveau
  dt_agg_y <- cbind(dt_vz, dt_y)
  dt_agg_y <- dt_agg_y[, lapply(.SD, function(x) sum(x * gew)), by=rdragnw]

  dt_agg_y <- dt_agg_y[, target := target - predictions][, .(rdragnw, target, gew)]
  dt_agg_y <- dt_agg_y[, target_per_persoon := target/gew][, .(rdragnw, target, gew,target_per_persoon)]
  #' Behoud de risicodragers met een geprognosticeerd financieel resultaat boven de boost_threshold of onder 
  #' -boost_threshold
  dt_agg_y <- dt_agg_y[target_per_persoon > boost_threshold | target_per_persoon < -boost_threshold ][, .(rdragnw, target)]
  print(paste('de gegevens van', nrow(dt_agg_y), 'risicodragers worden meegenomen'))
  dt_agg_y <- dt_agg_y[, .(rdragnw, target)]
  return(dt_agg_y)
}


i2i_train <- function(train_set,
                      alpha = 1,
                      lambda = 1e8,
                      correctie_versterking = 0.6,
                      boost_threshold = 2000,
                      base_model_source_path = 'src/models/VWS_Gupta_i2i_ML_XGB_v015.R')
{
  print('Dataprep:')
  
  lambda = as.double(lambda) # fread maakt er integer64 van als lambda groot is,
                             # maar ik weet niet of glmnet daar goed mee omgaat
  
  # Split train_set in features, target en verzekeraar
  dt_train_x <- train_set[, ..boost_model_features]
  dt_train_y <- train_set[, .(target)]
  dt_train_w <- train_set[, .(gew)]
  dt_train_vz <- train_set[, .(rdragnw)]
  dt_train_key <- train_set[, .(Pseudoniem_BSN, zvnr)]

  # Laad basismodel en voorspel op trainset
  base_model_env <- new.env()
  source(base_model_source_path, base_model_env)
  base_model <- base_model_env$i2i_train(train_set)
  base_model_preds <- base_model_env$i2i_predict(base_model, train_set[, -c('Pseudoniem_BSN', 'target', 'gew',
                                                                            'rdragnw', 'zvnr')])
  
  # Aggregeer hiermee target en selecteer verzekeraars om bandbreedte booster op te trainen
  dt_train_y <- prepareer_data_train_target(dt_train_y, dt_train_w, dt_train_vz, base_model_preds, boost_threshold)
  
  rdragnw_to_keep <- dt_train_y$rdragnw
  dt_train_y <- dt_train_y[rdragnw %in% rdragnw_to_keep][, target] # dt_train_y has already been aggregated

  # Prepareer features voor de geselecteerde verzekeraars
  dt_train_x <- prepareer_data_train_features(dt_train_x, dt_train_w, dt_train_vz, rdragnw_to_keep)
  
  # Train correctiemodel op geaggregeerde predicties van base model
  print('Train correctiemodel')
  boost_model <- glmnet(x = as.matrix(dt_train_x),
                        y = dt_train_y,
                        family = "gaussian",
                        alpha = alpha,
                        lambda = lambda,
                        intercept = FALSE)
  
  
  # Return het geheel van base model predicties file + correctiemodel en versterkingsfactor
  return(list(base_model_source_path = base_model_source_path,
              trained_base_model = base_model,
              boost_model = boost_model,
              correctie_versterking = correctie_versterking))
}

i2i_predict <- function(model, test_set){
  # Prepareer data
  print('Dataprep')
  dt_test_x <- test_set[, ..boost_model_features]

  # Laad basismodel en voorspel daarmee
  base_model_env <- new.env()
  source(model$base_model_source_path, base_model_env)
  base_model_preds <- base_model_env$i2i_predict(model$trained_base_model, test_set[, -c('Pseudoniem_BSN', 'zvnr')])
  
  #vanwege geheugenproblemen gooien we alvast wat stuff weg
  rm(list=ls(envir=base_model_env), envir=base_model_env)
  model$trained_base_model <- NULL
  gc()

  nrow <- nrow(dt_test_x)
  half <- round(nrow/2)
  dt_test_x_1 <- dt_test_x[1:half, ]
  dt_test_x_2 <- dt_test_x[(half+1):nrow, ]
  rm(dt_test_x);gc()
  
  # Voorspel correctie met boostmodel
  correctie_preds_1 <- predict(model$boost_model, as.matrix(one_hot_encode_features(dt_test_x_1)))
  rm(dt_test_x_1);gc()
  correctie_preds_2 <- predict(model$boost_model, as.matrix(one_hot_encode_features(dt_test_x_2)))
  rm(dt_test_x_2);gc()
  correctie_preds <- c(correctie_preds_1, correctie_preds_2)
  rm(correctie_preds_1, correctie_preds_2);gc()
  
  # Return gecorigeerde voorspelling
  return(as.vector(base_model_preds + model$correctie_versterking * correctie_preds))
}