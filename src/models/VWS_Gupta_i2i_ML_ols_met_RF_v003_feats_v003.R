#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Bestandsnaam: VWS_Gupta_i2i_ML_ols_met_rf_v003_feats_vxxx.R
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
  feats = c( 'lgnw' , 'regsom',   'avinw',   'sesnw', 'ppanw',  'pdkg',    'sdkg',  'fkg00',  'fkg01',
             'fkg02',  'fkg03',   'fkg04',   'fkg05', 'fkg06', 'fkg07',   'fkg08',  'fkg09',  'fkg10',
             'fkg11',  'fkg12',   'fkg13',   'fkg14', 'fkg15', 'fkg16',   'fkg17',  'fkg18',  'fkg19',
             'fkg20',  'fkg21',   'fkg22',   'fkg23', 'fkg24', 'fkg25',   'fkg26',  'fkg27',  'fkg28',
             'fkg29',  'fkg30',   'fkg31',   'fkg32', 'fkg33', 'fkg34',   'fkg35',  'fkg36',  'fkg37',
             'mhk',    'hkg',     'fdg',     'mvv',   'leeftijd')
  if (bevat_gew_target)
      numeric_cols = c('target', 'gew')
  else
      numeric_cols = c()
  
  dt_set <- dt_set[, c(feats, numeric_cols), with = FALSE]
  
  # Invullen NA-leeftijd
  dt_set[is.na(leeftijd) & (lgnw == 01 | lgnw == 22), leeftijd := 0]
  dt_set[is.na(leeftijd) & (lgnw == 02 | lgnw == 23), leeftijd := 0.5]
  dt_set[is.na(leeftijd) & (lgnw == 03 | lgnw == 24), leeftijd := 2.5]
  dt_set[is.na(leeftijd) & (lgnw == 04 | lgnw == 25), leeftijd := 7]
  dt_set[is.na(leeftijd) & (lgnw == 05 | lgnw == 26), leeftijd := 12]
  dt_set[is.na(leeftijd) & (lgnw == 06 | lgnw == 27), leeftijd := 16]
  dt_set[is.na(leeftijd) & (lgnw == 07 | lgnw == 28), leeftijd := 21]
  dt_set[is.na(leeftijd) & (lgnw == 08 | lgnw == 29), leeftijd := 27]
  dt_set[is.na(leeftijd) & (lgnw == 09 | lgnw == 30), leeftijd := 32]
  dt_set[is.na(leeftijd) & (lgnw == 10 | lgnw == 31), leeftijd := 37]
  dt_set[is.na(leeftijd) & (lgnw == 11 | lgnw == 32), leeftijd := 42]
  dt_set[is.na(leeftijd) & (lgnw == 12 | lgnw == 33), leeftijd := 47]
  dt_set[is.na(leeftijd) & (lgnw == 13 | lgnw == 34), leeftijd := 52]
  dt_set[is.na(leeftijd) & (lgnw == 14 | lgnw == 35), leeftijd := 57]
  dt_set[is.na(leeftijd) & (lgnw == 15 | lgnw == 36), leeftijd := 62]
  dt_set[is.na(leeftijd) & (lgnw == 16 | lgnw == 37), leeftijd := 67]
  dt_set[is.na(leeftijd) & (lgnw == 17 | lgnw == 38), leeftijd := 72]
  dt_set[is.na(leeftijd) & (lgnw == 18 | lgnw == 39), leeftijd := 77]
  dt_set[is.na(leeftijd) & (lgnw == 19 | lgnw == 40), leeftijd := 82]
  dt_set[is.na(leeftijd) & (lgnw == 20 | lgnw == 41), leeftijd := 87]
  dt_set[is.na(leeftijd) & (lgnw == 21 | lgnw == 42), leeftijd := 92]
  
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
