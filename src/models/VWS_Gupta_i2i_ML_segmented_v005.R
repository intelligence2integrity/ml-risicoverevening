#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

#' For questions contact:

#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma: VWS_Gupta_i2i_ML_segmented_v003.R
#' Doel:      gewogen trainen gesegmenteerde lineaire regressie op (gedeelte) van data set OT2020
#'            voorspellen van (ander) deel van data set OT2020 
#' Auteurs:   Jules van Ligtenberg
#'            Diederik Perdok
#' Opmerking: Deze file bevat alleen de code die afhankelijk is van het gekozen model. Deze file sourcen en 
#'            (gedeeltelijk) uitvoeren (via bijvoorbeeld doe_cv).
#'            
########################################################################################################################

library(data.table)
library(partykit)
library(stringr)

i2i_features <-  c('avinw',      'fdg',    'fkg00',    'fkg01',    'fkg02',    'fkg03',    'fkg04',    'fkg05',
                   'fkg06',    'fkg07',    'fkg08',    'fkg09',    'fkg10',    'fkg11',    'fkg12',    'fkg13',
                   'fkg14',    'fkg15',    'fkg16',    'fkg17',    'fkg18',    'fkg19',    'fkg20',    'fkg21',
                   'fkg22',    'fkg23',    'fkg24',    'fkg25',    'fkg26',    'fkg27',    'fkg28',    'fkg29',
                   'fkg30',    'fkg31',    'fkg32',    'fkg33',    'fkg34',    'fkg35',    'fkg36',    'fkg37',
                     'hkg', 'leeftijd',     'lgnw',      'mhk',      'mvv',     'pdkg',    'ppanw',   'regsom',
                    'sdkg',     'sesnw')

prepareer_data <- function(data_set){
  # Selecteer alleen vereiste kolommen
  data_set <- data_set[, i2i_features, with = FALSE]
  
  # Kolom lgnw gebruiken om geslacht te bepalen, en onbekende continue leeftijd in te vullen
  data_set[, geslacht := 0]
  data_set[lgnw > 21, geslacht := 1]
  
  data_set[is.na(leeftijd) & (lgnw == 01 | lgnw == 22), leeftijd := 0]
  data_set[is.na(leeftijd) & (lgnw == 02 | lgnw == 23), leeftijd := 0.5]
  data_set[is.na(leeftijd) & (lgnw == 03 | lgnw == 24), leeftijd := 2.5]
  data_set[is.na(leeftijd) & (lgnw == 04 | lgnw == 25), leeftijd := 7]
  data_set[is.na(leeftijd) & (lgnw == 05 | lgnw == 26), leeftijd := 12]
  data_set[is.na(leeftijd) & (lgnw == 06 | lgnw == 27), leeftijd := 16]
  data_set[is.na(leeftijd) & (lgnw == 07 | lgnw == 28), leeftijd := 21]
  data_set[is.na(leeftijd) & (lgnw == 08 | lgnw == 29), leeftijd := 27]
  data_set[is.na(leeftijd) & (lgnw == 09 | lgnw == 30), leeftijd := 32]
  data_set[is.na(leeftijd) & (lgnw == 10 | lgnw == 31), leeftijd := 37]
  data_set[is.na(leeftijd) & (lgnw == 11 | lgnw == 32), leeftijd := 42]
  data_set[is.na(leeftijd) & (lgnw == 12 | lgnw == 33), leeftijd := 47]
  data_set[is.na(leeftijd) & (lgnw == 13 | lgnw == 34), leeftijd := 52]
  data_set[is.na(leeftijd) & (lgnw == 14 | lgnw == 35), leeftijd := 57]
  data_set[is.na(leeftijd) & (lgnw == 15 | lgnw == 36), leeftijd := 62]
  data_set[is.na(leeftijd) & (lgnw == 16 | lgnw == 37), leeftijd := 67]
  data_set[is.na(leeftijd) & (lgnw == 17 | lgnw == 38), leeftijd := 72]
  data_set[is.na(leeftijd) & (lgnw == 18 | lgnw == 39), leeftijd := 77]
  data_set[is.na(leeftijd) & (lgnw == 19 | lgnw == 40), leeftijd := 82]
  data_set[is.na(leeftijd) & (lgnw == 20 | lgnw == 41), leeftijd := 87]
  data_set[is.na(leeftijd) & (lgnw == 21 | lgnw == 42), leeftijd := 92]
  data_set[, lgnw := NULL]
  
  # Peuter kolommen avinw, ppanw en sesnw los van leeftijd
  data_set[avinw %in% 01:06, avi_los := 0] # Duurzaam en volledig arbeidsongeschikten (IVA)
  data_set[avinw %in% 07:12, avi_los := 1] # Arbeidsongeschikten excl. IVA
  data_set[avinw %in% 13:18, avi_los := 2] # Bijstandsgerechtigden
  data_set[avinw %in% 19:20, avi_los := 3] # Studenten
  data_set[avinw %in% 25:30, avi_los := 4] # Zelfstandigen
  data_set[avinw %in% 31:33, avi_los := 5] # Hoogopgeleiden
  data_set[avinw %in% 37:42, avi_los := 6] # Referentiegroep
  data_set[avinw == 0, avi_los := 6]       # 70+ jaar
  data_set[, avinw := NULL]
  
  data_set[sesnw %in% 01:03, ses_los := 0] # SES zeer laag
  data_set[sesnw %in% 04:06, ses_los := 1] # SES laag
  data_set[sesnw %in% 07:09, ses_los := 2] # SES midden
  data_set[sesnw %in% 10:12, ses_los := 3] # SES hoog
  data_set[, sesnw := NULL]
  
  data_set[ppanw %in% 01:03, ppa_los := 0] # Wlz-instelling, blijvend
  data_set[ppanw %in% 04:06, ppa_los := 1] # Wlz-instelling, instromend
  data_set[ppanw %in% 07:09, ppa_los := 2] # Eenpersoonshuishouden
  data_set[ppanw %in% 10:12, ppa_los := 3] # Overig
  data_set[ppanw == 0, ppa_los := 3]       # 0-17 jaar
  data_set[, ppanw := NULL]
  
  # One-hot encoding (werkt ook als niet alle mogelijke waarden aanwezig zijn in de trainset)
  one_hot_encoding(data_set, 'regsom',  1:10) # functie afkomstig uit VWS_Gupta_i2i_ML_utils_vxxx.R; moet reeds gesourced
  one_hot_encoding(data_set, 'ses_los', 0:3)  # zijn, bijvoorbeeld in VWS_Gupta_i2i_ML_doe_cross_validation_vxxx.R
  one_hot_encoding(data_set, 'ppa_los', 0:3)
  one_hot_encoding(data_set, 'pdkg',    0:15)
  one_hot_encoding(data_set, 'sdkg',    0:7)
  one_hot_encoding(data_set, 'avi_los', 0:6)
  one_hot_encoding(data_set, 'mhk',     0:8)
  one_hot_encoding(data_set, 'hkg',     0:10)
  one_hot_encoding(data_set, 'fdg',     0:4)
  one_hot_encoding(data_set, 'mvv',     0:9)
  
  return(data_set)
}

# Aanpak: Maak een decision tree o.b.v. leeftijd en train een ols-model voor elke leaf-node. Het R-package partykit
# doet precies dit, maar probeert onderweg alle mogelijke leeftijdssplits, en vereist dust het trainen van ongeveer
# aantal_leeftijden * boomdiepte OLS-modellen. Om het trainen te versnellen maken we eerst een partykit-model op een
# subset van de data waarmee we de boomstructuur vaststellen. Hertrain vervolgens het OLS-model in elke leaf opnieuw
# op de volledige trainingsset.
i2i_train <- function(train_set, verbose = 0){
  target <- train_set$target
  gew <- train_set$gew
  dt_train <- prepareer_data(train_set[, -c('target', 'gew')])
  rm(train_set)
  
  subset_idxs <- sample(1:nrow(dt_train), round(nrow(dt_train) * 0.01))
  target_subset <- target[subset_idxs]
  gew_subset <- gew[subset_idxs]
  dt_train_subset <- dt_train[subset_idxs]
  
  # Voeg ruis toe
  # Partykit heeft wat problemen met numerieke stabiliteit: de "parameter instability test" die onderdeel is van het
  # gebruikte algoritme mislukt, tenzij we een beetje ruis optellen bij de 0/1-kolommen. Dit doen we alleen
  # tijdens het trainen van het partykit model; niet bij het hertrainen van de ols modellen in de leave-nodes.
  bin_kolommen <- Filter(function(x) startsWith(x, 'fkg'), names(dt_train))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'regsom'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'ses_los'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'ppa_los'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'pdkg'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'sdkg'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'avi_los'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'mhk'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'hkg'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'fdg'), names(dt_train)))
  bin_kolommen <- c(bin_kolommen, Filter(function(x) startsWith(x, 'mvv'), names(dt_train)))
  
  dt_train_subset[, c(bin_kolommen) := lapply(.SD, function(col) col + rnorm(length(col), mean = 0, sd = 1e-5)),
                  .SDcols = bin_kolommen]

  # train model op subset
  prev_opt <- getOption('warnPartialMatchDollar')
  options(warnPartialMatchDollar = F) # moet FALSE zijn voor partykit i.v.m. omdat wij warnings ophogen naar errors
  model_subset <- lmtree(target_subset ~ . | leeftijd,
                         data = dt_train_subset,
                         verbose = T,
                         weights = gew_subset,
                         minsize = 1e4)
  options(warnPartialMatchDollar = prev_opt)
  
  # Leeftijdsbuckets extraheren uit het getrainde model
  # https://stats.stackexchange.com/questions/354066/how-to-extract-the-split-points-of-mob
  nodes <- nodeids(model_subset)
  if (length(nodes) > 1){
    nodes_leaf <- nodeids(model_subset, terminal = TRUE)
    nodes_intern <- nodes[!nodes %in% nodes_leaf]
    grenzen <- sapply(nodes_intern, function(i) split_node(node_party(model_subset[[i]]))$breaks)
    grenzen <- sort(grenzen)
    
    secties <- vector(mode = 'list', length = length(grenzen) + 1) # lijst van (ondergrens, bovengrens] intervallen
    secties[[1]] = c(-1, 0)
    for (i in 1:length(grenzen)){
      secties[[i]][[2]] <- grenzen[[i]]
      secties[[i + 1]] = c(grenzen[[i]], 0)
    }
    secties[[length(secties)]][[2]] <- Inf
  } else {
    secties = list(c(-1, Inf)) # Als er niet is gesegmenteerd door partykit, dan gewoon alles in 1
  }
  print(paste('klein model klaar;', length(secties), 'leeftijdscategoriën gemaakt met secties:',
              paste0(secties, collapse = '; ')))
  
  # OLS-model in elke leaf hertrainen op volledige dataset
  model_definitief <- vector(mode = 'list', length = length(secties))
  attr(model_definitief, 'secties') <- secties
  for (i in 1:length(secties)){
    sectie <- secties[[i]]
    mask <- (dt_train$leeftijd >= sectie[[1]]) & (dt_train$leeftijd < sectie[[2]])
    print(paste('Aantal rijen voor trainen lm-model in leaf', i, ':', nrow(dt_train[mask])))
    model_definitief[[i]] <- lm(target[mask] ~ ., data = dt_train[mask], weights = gew[mask], model = F)

    # voorkomen dat gehele dataset onderdeel wordt van model (geeft enorme filesize bij opslaan)
    attr(model_definitief[[i]]$terms, ".Environment") <- NULL
    model_definitief[[i]]$qr$qr <- NULL
  }
  
  return(model_definitief)
}

i2i_predict <- function(model, test_set){
  dt_test <- prepareer_data(test_set)
  
  pred <- rep(-1e10, nrow(dt_test)) # zodat we het meteen zien als er een predictie niet wordt ingevuld
  secties <- attr(model, 'secties')
  vorig_opt <- getOption('warn')
  options(warn = 0) # Mag niet crashen tijdens debug op 'prediction from a rank-deficient fit may be misleading'
  for (i in 1:length(secties)) {
    mask <- (dt_test$leeftijd > secties[[i]][[1]]) & (dt_test$leeftijd <= secties[[i]][[2]])
    pred[mask] <- predict(model[[i]], dt_test[mask])
  }
  options(warn = vorig_opt)
  
  return(pred)
}
