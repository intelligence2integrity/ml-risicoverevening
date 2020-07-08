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
#'               s-during-development.
########################################################################################################################

library(data.table)
library(keras)
library(parallel)
library(stringr)

prepareer_data_features <- function(dt_x, means = NULL, stdevs = NULL){
  # Selecteer gebruikte features
  print('Selecteer features')
  dx_feats = c('dx_groep_101018', 'dx_groep_101072', 'dx_groep_101150', 'dx_groep_101171', 'dx_groep_102019',
               'dx_groep_111048', 'dx_groep_111049', 'dx_groep_111050', 'dx_groep_112007', 'dx_groep_113171',
               'dx_groep_113172', 'dx_groep_151007', 'dx_groep_151016', 'dx_groep_16215' , 'dx_groep_1731'  ,
               'dx_groep_1732'  , 'dx_groep_1733'  , 'dx_groep_1741'  , 'dx_groep_1742'  , 'dx_groep_1743'  ,
               'dx_groep_175'   , 'dx_groep_1750'  , 'dx_groep_176'   , 'dx_groep_177'   , 'dx_groep_178'   ,
               'dx_groep_179'   , 'dx_groep_1801'  , 'dx_groep_1802'  , 'dx_groep_21003' , 'dx_groep_21007' ,
               'dx_groep_21008' , 'dx_groep_21009' , 'dx_groep_21010' , 'dx_groep_21011' , 'dx_groep_21013' ,
               'dx_groep_21014' , 'dx_groep_21015' , 'dx_groep_21016' , 'dx_groep_21017' , 'dx_groep_21018' ,
               'dx_groep_21019' , 'dx_groep_21020' , 'dx_groep_21021' , 'dx_groep_21022' , 'dx_groep_21027' ,
               'dx_groep_21032' , 'dx_groep_21033' , 'dx_groep_21034' , 'dx_groep_21035' , 'dx_groep_21036' ,
               'dx_groep_21037' , 'dx_groep_21041' , 'dx_groep_21043' , 'dx_groep_21048' , 'dx_groep_21049' ,
               'dx_groep_21055' , 'dx_groep_21056' , 'dx_groep_21057' , 'dx_groep_21077' , 'dx_groep_21081' ,
               'dx_groep_21084' , 'dx_groep_21086' , 'dx_groep_21089' , 'dx_groep_21093' , 'dx_groep_21096' ,
               'dx_groep_21097' , 'dx_groep_21098' , 'dx_groep_21099' , 'dx_groep_211007', 'dx_groep_211048',
               'dx_groep_21105' , 'dx_groep_211070', 'dx_groep_211072', 'dx_groep_211073', 'dx_groep_211093',
               'dx_groep_211144', 'dx_groep_21115' , 'dx_groep_211171', 'dx_groep_222134', 'dx_groep_231008',
               'dx_groep_23171' , 'dx_groep_232007', 'dx_groep_233171', 'dx_groep_233371', 'dx_groep_243021',
               'dx_groep_31077' , 'dx_groep_31080' , 'dx_groep_31081' , 'dx_groep_31084' , 'dx_groep_31085' ,
               'dx_groep_31086' , 'dx_groep_31087' , 'dx_groep_31089' , 'dx_groep_31097' , 'dx_groep_31171' ,
               'dx_groep_331070', 'dx_groep_331089', 'dx_groep_331093', 'dx_groep_333025', 'dx_groep_400001',
               'dx_groep_400002', 'dx_groep_400003', 'dx_groep_400004', 'dx_groep_400006', 'dx_groep_400007',
               'dx_groep_400008', 'dx_groep_41007' , 'dx_groep_41013' , 'dx_groep_41048' , 'dx_groep_41097' ,
               'dx_groep_41105' , 'dx_groep_41110' , 'dx_groep_41111' , 'dx_groep_42089' , 'dx_groep_500001',
               'dx_groep_500002', 'dx_groep_500003', 'dx_groep_500004', 'dx_groep_500005', 'dx_groep_500006',
               'dx_groep_500007', 'dx_groep_500008', 'dx_groep_500009', 'dx_groep_500010', 'dx_groep_500011',
               'dx_groep_500012', 'dx_groep_500013', 'dx_groep_500014', 'dx_groep_500015', 'dx_groep_500016',
               'dx_groep_500019', 'dx_groep_500020', 'dx_groep_500021', 'dx_groep_500022', 'dx_groep_500023',
               'dx_groep_500024', 'dx_groep_500025', 'dx_groep_500026', 'dx_groep_500027', 'dx_groep_500028',
               'dx_groep_500029', 'dx_groep_500030', 'dx_groep_500031', 'dx_groep_500032', 'dx_groep_500033',
               'dx_groep_500034', 'dx_groep_500035', 'dx_groep_500036', 'dx_groep_51048' , 'dx_groep_51049' ,
               'dx_groep_51050' , 'dx_groep_53098' , 'dx_groep_600010', 'dx_groep_600020', 'dx_groep_600021',
               'dx_groep_600030', 'dx_groep_600040', 'dx_groep_600050', 'dx_groep_71009' , 'dx_groep_71010' ,
               'dx_groep_71011' , 'dx_groep_71025' , 'dx_groep_71032' , 'dx_groep_71034' , 'dx_groep_71035' ,
               'dx_groep_71036' , 'dx_groep_71037' , 'dx_groep_71041' , 'dx_groep_71098' , 'dx_groep_72007' ,
               'dx_groep_83070' , 'dx_groep_91009' , 'dx_groep_91010' , 'dx_groep_91011' , 'dx_groep_91013' ,
               'dx_groep_91014' , 'dx_groep_91015' , 'dx_groep_91021' , 'dx_groep_91027' , 'dx_groep_91032' ,
               'dx_groep_91036' , 'dx_groep_91037' , 'dx_groep_91041' , 'dx_groep_91043' , 'dx_groep_91049' ,
               'dx_groep_91070' , 'dx_groep_91092' , 'dx_groep_91095' , 'dx_groep_91096' , 'dx_groep_91097' ,
               'dx_groep_91134' , 'dx_groep_91150' , 'dx_groep_91151' , 'dx_groep_92007' , 'dx_groep_92153' ,
               'dx_groep_93049' , 'dx_groep_93050' , 'dx_groep_93051')
  
  fkg_continu_feats = c('aantal_fkg_01', 'aantal_fkg_02', 'aantal_fkg_03', 'aantal_fkg_04', 'aantal_fkg_05',
                        'aantal_fkg_06', 'aantal_fkg_07', 'aantal_fkg_08', 'aantal_fkg_09', 'aantal_fkg_10',
                        'aantal_fkg_11', 'aantal_fkg_12', 'aantal_fkg_13', 'aantal_fkg_14', 'aantal_fkg_16',
                        'aantal_fkg_17', 'aantal_fkg_18', 'aantal_fkg_19', 'aantal_fkg_20', 'aantal_fkg_21',
                        'aantal_fkg_22', 'aantal_fkg_23', 'aantal_fkg_24', 'aantal_fkg_25', 'aantal_fkg_26',
                        'aantal_fkg_27', 'aantal_fkg_28', 'aantal_fkg_29', 'aantal_fkg_30', 'aantal_fkg_31',
                        'aantal_fkg_32', 'aantal_fkg_33', 'aantal_fkg_34', 'aantal_fkg_35', 'aantal_fkg_36',
                        'aantal_fkg_37')
  
  andere_feats = c('aantal_diag',    'aantal_spec',    'aantal_zp',      'avinw', 'fdg',   'fkg00', 'fkg01', 'fkg02',
                   'fkg03', 'fkg04', 'fkg05', 'fkg06', 'fkg07', 'fkg08', 'fkg09', 'fkg10', 'fkg11', 'fkg12', 'fkg13',
                   'fkg14', 'fkg15', 'fkg16', 'fkg17', 'fkg18', 'fkg19', 'fkg20', 'fkg21', 'fkg22', 'fkg23', 'fkg24',
                   'fkg25', 'fkg26', 'fkg27', 'fkg28', 'fkg29', 'fkg30', 'fkg31', 'fkg32', 'fkg33', 'fkg34', 'fkg35',
                   'fkg36', 'fkg37', 'hkg',   'lgnw',  'mhk',   'mvv',   'n_verpleeg',     'n_operatief',    'ppanw',
                   'pdkg',  'regsom','sdkg',  'sesnw')
  
  dt_x <- dt_x[, c(andere_feats, dx_feats, fkg_continu_feats), with = FALSE]
  
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
  if (is.null(means))
    print('Bereken means')
    means <- dt_x[, lapply(.SD, mean)]
  if (is.null(stdevs))
    print('Bereken stdevs')
    stdevs <- dt_x[, lapply(.SD, sd)]
  
  print('Normaliseer')
  for (i in 1:ncol(dt_x))
    dt_x[, names(dt_x)[i] := (get(names(dt_x)[i]) - means[[i]]) / (stdevs[[i]] + 1e-8)]
  
  gc(full = T)
  
  return(list('dt_x' = dt_x, 'means' = means, 'stdevs' = stdevs))
}

definieer_en_train_model <- function(dt_train_x, dt_train_y, dt_train_w){
  # Model definieren
  print('Definieer model')
  model <- keras_model_sequential() %>%
    layer_dense(units = 512, activation = 'relu', input_shape = ncol(dt_train_x)) %>%
    layer_dropout(0.35, seed = 0) %>%
    layer_dense(units = 256, activation = 'relu') %>%
    layer_dropout(0.35, seed = 0) %>%
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
      batch_size = 2048,
      epochs = 45,
      validation_split = 0,
      verbose = 1,
      shuffle = T,
      sample_weight = dt_train_w)
  
  return (model)
}

i2i_train <- function(train_set, verbose = 0){
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
  return(as.vector(predict(model, as.matrix(geprepareerd$dt_x))))
}
