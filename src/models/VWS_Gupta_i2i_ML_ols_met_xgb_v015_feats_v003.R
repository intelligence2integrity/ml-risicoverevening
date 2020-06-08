#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Bestandsnaam: VWS_Gupta_i2i_ML_ols_met_xgb_v015_feats_vxxx.R
#' Doel:         gewogen trainen OLS model zoals in WOR973 op (gedeelte) van data set OT2020
#'               voorspellen van (ander) deel van data set OT2020 
#' Auteurs:      Jules van Ligtenberg
#'               Diederik Perdok
#' Opmerking:    Deze file bevat alleen de code die afhankelijk is van het gekozen model. Deze file sourcen en 
#'               (gedeeltelijk) uitvoeren (via bijvoorbeeld doe_cv).
#' Historie:
#' Versie  R2-OOF-score
#'      5  ?
########################################################################################################################

library(data.table)

########################################################################################################################
# genereer_constrained_formule_termen
########################################################################################################################
# Gegeven een set features, en een "= 0 constraint" op de coefficiënten van deze features, return de
# rechterkant van een R-formule (als string) waarin deze constraint is verwerkt. Dit gebeurt door
# het elimineren van de laatste feature.
#
# Voorbeeld:
# Er is een model "Y = beta1 * x1 + beta2 * x2 + beta3 * x3 + beta4 * x4 + .... + eps".
# Voor features x1 t/m x4 geldt een constraint: "2 * beta1 + 2 * beta2 + 1 * beta3 + 3 * beta4 = 0".
#
# M.b.v. de constraint kan het model worden geschreven als
# "Y = beta1 * x1 + beta2 * x2 + beta3 * x3 + ((- 2 * beta1 - beta3 - 3 * beta4) / 3) * x4 + .... + eps", wat
# gelijk is aan "Y = beta1 * (x1 - 2/3 * x4) + beta2 * (x2 - 2/3 * x4) + beta3 * (x3 - 1/3 * x4) + ... + eps".
# Het "normbedrag" voor x1 is dan gelijk aan beta1.
# Geschreven als R-formule: "Y ~ I(x1 - 2/3 * x4) + I(x2 - 2/3 * x4) + I(x3 - 1/3 * x4) + ... - 1".
#
# Dus met argumenten feats = c('x1', 'x2', 'x3', 'x4') en constraint_coeff = c(2, 2, 1, 3), wordt de returnwaarde
# '+ I(x1 - 2/3 * x4) + I(x2 - 2/3 * x4) + I(x3 - 1/3 * x4)'.

genereer_constrained_formule_termen <- function(feats, constraint_coeff) {
  formule <- ''
  for (idx in 1:(length(feats) - 1)) {
    formule <- paste0(formule,
                     ' + I(',
                     feats[[idx]],
                     ' + ',
                     -constraint_coeff[[idx]] / constraint_coeff[[length(constraint_coeff)]],
                     ' * ',
                     feats[[length(feats)]],
                     ')')
  }
  
  return (formule)
}

########################################################################################################################
# prepareer_data
########################################################################################################################
# Alle vereiste preprocessing zoals one-hot-encoding. Wordt zowel op train als op test toegepast.

prepareer_data <- function(dt_set, bevat_gew_target){
  # definieer features en gooi andere kolommen weg
  feats = c('aantal_fkg_01'  , 'aantal_fkg_02'  , 'aantal_fkg_03'  , 'aantal_fkg_04'  , 'aantal_fkg_05'  ,
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
  
  if (bevat_gew_target)
      numeric_cols = c('target', 'gew')
  else
      numeric_cols = c()
  
  dt_set <- dt_set[, c(feats, numeric_cols), with = FALSE]
  
  # one-hot encoding (werkt ook als niet alle mogelijke waarden aanwezig zijn in de trainset)
  one_hot_encoding(dt_set, 'lgnw',   1:42)
  one_hot_encoding(dt_set, 'regsom', 1:10)
  one_hot_encoding(dt_set, 'sesnw',  1:12)
  one_hot_encoding(dt_set, 'ppanw',  0:15)
  one_hot_encoding(dt_set, 'pdkg',   0:15)
  one_hot_encoding(dt_set, 'sdkg',   0:7)
  one_hot_encoding(dt_set, 'avinw',  0:42)
  one_hot_encoding(dt_set, 'mhk',    0:8)
  one_hot_encoding(dt_set, 'hkg',    0:10)
  one_hot_encoding(dt_set, 'fdg',    0:4)
  one_hot_encoding(dt_set, 'mvv',    0:9)
  
  # overig
  dt_set$avinw_0 <- NULL
  dt_set$ppanw_0 <- NULL # Verwijder features die op 0 gezet worden door WOR
  if (bevat_gew_target)
    dt_set[, target := target / 100] # eurocenten naar euro's
  
  return (dt_set)
}

########################################################################################################################
# i2i_train
########################################################################################################################

i2i_train <- function(train_set){
  dt_train = prepareer_data(train_set, bevat_gew_target = T)
  
  # Bouw formule op, door het verwerken van de equality constraints via het elimineren van normbedragen. Zo verkrijgen
  # we dezelfde normbedragen als in het officiële SAS model, al hebben we per constraint 1 bedrag minder dat
  # desgewenst is terug te rekenen. Het is in (base-)R niet mogelijk om constraints direct toe te passen in een lineair
  # model, en ik heb geen efficiënt package gevonden waarmee dit mogelijk is.
  feats <- colnames(dt_train[, -c('target', 'gew')])
  constraint_cols <- list(feats[startsWith(feats, 'lgnw')], # 1 constraint voor elk lijstelement
                          feats[startsWith(feats, 'fkg')],  # lgnw's moeten in het eerste lijstelement
                          feats[startsWith(feats, 'pdkg')],
                          feats[startsWith(feats, 'sdkg')],
                          feats[startsWith(feats, 'lgnw')],
                          feats[startsWith(feats, 'hkg')],
                          feats[startsWith(feats, 'regsom')],
                          feats[startsWith(feats, 'mhk')],
                          feats[startsWith(feats, 'fdg')],
                          feats[startsWith(feats, 'mvv')],
                          c('avinw_1', 'avinw_7',  'avinw_13', 'avinw_19', 'avinw_25', 'avinw_31', 'avinw_37'),
                          c('avinw_2', 'avinw_8',  'avinw_14', 'avinw_20', 'avinw_26', 'avinw_32', 'avinw_38'),
                          c('avinw_3', 'avinw_9',  'avinw_15',             'avinw_27', 'avinw_33', 'avinw_39'),
                          c('avinw_4', 'avinw_10', 'avinw_16',             'avinw_28',             'avinw_40'),
                          c('avinw_5', 'avinw_11', 'avinw_17',             'avinw_29',             'avinw_41'),
                          c('avinw_6', 'avinw_12', 'avinw_18',             'avinw_30',             'avinw_42'),
                          c('sesnw_1', 'sesnw_4', 'sesnw_7', 'sesnw_10'),
                          c('sesnw_2', 'sesnw_5', 'sesnw_8', 'sesnw_11'),
                          c('sesnw_3', 'sesnw_6', 'sesnw_9', 'sesnw_12'),
                          c('ppanw_1', 'ppanw_4', 'ppanw_7', 'ppanw_10'),
                          c('ppanw_2', 'ppanw_5', 'ppanw_8', 'ppanw_11'),
                          c('ppanw_3', 'ppanw_6', 'ppanw_9', 'ppanw_12'))
  
  for (i in 1:length(constraint_cols)){
    cols = constraint_cols[[i]]
    constraint_cols[[i]] <- cols[order(nchar(cols), cols)]  # sorteer prefix_2 voor prefix_10
  }
  
  formule_rhs <- ''
  for (cols in constraint_cols){
    constraint_coeff <- c(rep(0, length(cols)))
    
    for (i in 1:length(cols)){
      col_name = cols[[i]]
      constraint_coeff[[i]] <- dt_train[, sum(get(col_name) * gew)]
    }
    
    formule_rhs <- paste0(formule_rhs, genereer_constrained_formule_termen(cols, constraint_coeff))
  }
  formule_rhs <- paste0(formule_rhs, ' + ', paste0(setdiff(feats, unlist(constraint_cols)), collapse = ' + '))
  
  max_lgnw <- constraint_cols[[1]][[length(constraint_cols[[1]])]]
  sum_target <- dt_train[, sum(target * gew)] # De lgnw-features sommeren niet tot 0, maar tot sum_target
  linkerkant_correctie = sum_target / dt_train[, sum(get(max_lgnw) * gew),] # Dat wijzigt de linkerkant van de formule
  formule_lhs <- paste0('I(target + ', - linkerkant_correctie, ' * ', max_lgnw, ')')
  formule <- paste0(formule_lhs, ' ~ ', formule_rhs, ' - 1') # -1 om de intercept uit te zetten)
  
  # train het model
  model <- lm(formula(formule), weights = gew, data = dt_train)
  model$linkerkant_correctie <- linkerkant_correctie # nodig voor predictie van hoogste leeftijdscategorie
  model$max_lgnw <- max_lgnw # nodig voor predictie van hoogste leeftijdscategorie
  return (model)
}

########################################################################################################################
# i2i_predict
########################################################################################################################

i2i_predict <- function(model, test_set){
  dt_test = prepareer_data(test_set, bevat_gew_target = F)
  pred <- predict(model, newdata = dt_test)
  mask = dt_test[, get(model$max_lgnw) == 1]
  pred[mask] = pred[mask] + model$linkerkant_correctie
  pred <- round(pred * 100, 0) # Terug naar eurocenten omdat onze metric dit vereist. Afronden op centen doet WOR ook.
  
  return(pred)
}
