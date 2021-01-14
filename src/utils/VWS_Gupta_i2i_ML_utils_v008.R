########################################################################################################################
#' Programma:   VWS_Gupta_i2i_ML_utils_vXXX.R
#' Doel:        'utilities' die gebruikt worden door meerdere modellen en/of hulprogramma's
#' Auteurs:     Jules van Ligtenberg
#'              Diederik Perdok
#' versie 8:    vanaf versie 8 hebben we de naamgeving van (gewogen) bandbreedte gelijkgetrokken met eerder
#'              gepubliceerde rapporten. (dwz dat bandbreedte wordt gedefinieerd als het verschil van het grootste min
#'              het kleinste financieel resultaat PER VERZEKERDE van de verschillende risicodragers)
########################################################################################################################

library(data.table)

########################################################################################################################
# bandbreedte
########################################################################################################################
#' Doel: Bandbreedte van alle verzekeraars uitrekenen zoals in WOR 871 (financieel restultaat grootste - kleinste)
#' Parameters:
#'    Naam        Type       Functie
#'    y           integer    de actuele waarden van de target (in eurocenten)
#'    y_hat       integer    de voorspelling van de actuele waarde (in eurocenten)
#'    w           double     het gewicht van de voorspelling (typisch een waarde tussen de 1/365 en 1)
#'    verzekeraar integer    identificeert de verzekeraar (risicodrager)
#'    verbose     boolean    wel/niet printen informatie over berekening
#' returnwaarde: de bandbreedte van alle verzekeraars
bandbreedte <- function(y, y_hat, w, verzekeraar, verbose=F){
  if(verbose){
    if(length(unique(y)) < length(unique(y_hat))) cat("Warning: are you sure you didn't switch y and y_hat?\n")
  } 
  dt <- bandbreedte_01(y, y_hat, w, verzekeraar)
  if(verbose){
    cat('\n[bandbreedte] details bij berekening over', length(y), 'observaties:\n')
    print(dt[order(dt$financieel_resultaat), ])
  }
  return(dt[, max(financieel_resultaat) - min(financieel_resultaat)])
}

########################################################################################################################
# bandbreedte_01
########################################################################################################################
#' Doel: hulpfunctie voor bandbreedte en bandbreedte_rond_nul
#' Parameters:
#'    Naam        Type       Functie
#'    y           integer    de actuele waarden van de target (in eurocenten)
#'    y_hat       integer    de voorspelling van de actuele waarde (in eurocenten)
#'    w           double     het gewicht van de voorspelling (typisch een waarde tussen de 1/365 en 1)
#'    verzekeraar integer    identificeert de verzekeraar (risicodrager)
#' returnwaarde: de bandbreedte van alle verzekeraars
bandbreedte_01 <- function(y, y_hat, w, verzekeraar){
  stopifnot(length(y) == length(y_hat))
  stopifnot(length(y) == length(w))
  stopifnot(length(y) == length(verzekeraar))
  dt <- data.table(financieel_resultaat = (y_hat - y), w=w, verzekeraar = verzekeraar)
  dt <- dt[, .(financieel_resultaat = sum(financieel_resultaat * w) / sum(w)), by = verzekeraar]
  return(dt)
}


########################################################################################################################
# bandbreedte_rond_nul
########################################################################################################################
#' Doel: controleer of de bandbreedte rond nul is op de onderhavige test/validatie set
#' Parameters:
#'    Naam        Type       Functie
#'    y           integer    de actuele waarden van de target (in eurocenten)
#'    y_hat       integer    de voorspelling van de actuele waarde (in eurocenten)
#'    w           double     het gewicht van de voorspelling (typisch een waarde tussen de 1/365 en 1)
#'    verzekeraar integer    identificeert de verzekeraar (risicodrager)
#'    verbose     boolean    wel/niet printen informatie over berekening
#' returnwaarde: zijn de waarden die de bandbreedte bepalen rond nul
bandbreedte_rond_nul <- function(y, y_hat, w, verzekeraar, verbose=F){
  dt <- bandbreedte_01(y, y_hat, w, verzekeraar)
  rond_nul <- max(dt$financieel_resultaat) > 0 && min(dt$financieel_resultaat) < 0
  if(verbose){
    if(!rond_nul){
      cat('\n[bandbreedte_rond_nul] is niet rond nul bij berekening over', length(y), 'observaties:\n')
      print(dt[order(dt$financieel_resultaat), ])
    }
  }
  return(rond_nul)
}

########################################################################################################################
# freeze_and_source
########################################################################################################################
#' Doel: de code vastzetten voor gebruik zodat later e.e.a. te reproduceren is
#' Preconditie: het bestand moet bestaan en de huidige gebruiker moet geautoriseerd zijn om de permissies van het
#'              bestand te veranderen
#' Parameters:
#'    Naam      Type      Functie
#'    file_name character de naam van de file
#' returnwaarde: geen
#'    

freeze_and_source <- function(file_name){
  Sys.chmod(file_name, mode = "0444", use_umask = T)
  source(file_name)
}

########################################################################################################################
# one_hot_encoding
########################################################################################################################
#' Doel: one-hot-encoding realiseren voor een specifieke variabele
#' Preconditie: Er komen in de data.table <dt> in veld <field> geen waarden voor die niet in values staan
#' Parameters:
#'    Naam   Type       Functie
#'    dt     data.table train of test set die aangepast moet worden
#'    field  character  varabele die gebruikt moet worden voor de one-hot-encoding
#'    values vector     voorgedefinieerde waarden die gebruikt gaan worden
#' returnwaarde: de aangepaste data, hierbij is de originele variabele <field> verwijderd
#'    
one_hot_encoding <- function(dt, field, values){
  extra_values <- setdiff(unique(dt[[field]]), values)
  if(length(extra_values) > 0){
    stop(paste('Er zijn onvoorziene extra waarden gevonden', paste(extra_values, collapse=' ')))
  }
  for(value in values){
    f <- paste0(field, '_', value)
    dt[, (f) := 0L]
    dt[get(field) == value, (f) := 1L]
  }
  dt[, (field) := NULL]
}

########################################################################################################################
# permutation_feature_importance
########################################################################################################################
#' Doel: Berekenen permutation feature importance voor een of meerdere features van een specifiek model
#' Preconditie: Het model moet getraind zijn
#' Parameters:
#'    Naam            Beschrijving
#'    model           Model dat aan i2i_predict kan worden meegegeven.
#'    i2i_predict     Functie met argumenten model en data_set; geeft vector van predicties terug.
#'    data_set        Data.frame met target, gew en alle features die het model verwacht.
#'    features        Character vector met featurenamen waarvan importance berekend moet worden. NULL voor alles.
#'
#' returnwaarde: data.frame met kolommen feature en importance
permutation_feature_importance <- function(model, i2i_predict, data_set, features = NULL, verbose = F){
  if (is.null(features))
    features = setdiff(names(data_set), c('gew', 'target'))
  
  if (verbose) cat('Bepalen R2 zonder permutaties\n')
  target <- data_set[['target']]
  gew <- data_set[['gew']]
  base_R2 <- R2(target, i2i_predict(model, data_set), gew)
  
  feat_imp <- data.frame(feature = character(length(features)), importance = numeric(length(features)))
  row <- 1
  for (feat in features){
    if(verbose) cat(paste0('Bezig met feature ', row, '/', length(features), '\n'))
    old_column <- data_set[[feat]]
    
    data_set[[feat]] = sample(old_column, length(old_column), replace = F)
    imp <- base_R2 - R2(target, i2i_predict(model, data_set), gew)
    feat_imp[row, 'feature'] <- feat
    feat_imp[row, 'importance'] <- imp
    
    data_set[[feat]] = old_column
    row <- row + 1
  }
  
  return(feat_imp)
}

########################################################################################################################
# R2
########################################################################################################################
#' Doel: R-squared uitrekenen met weging van speciale gevallen en gewogen gemiddelde van y 
#' Parameters:
#'    Naam   Type       Functie
#'    y      integer    de actuele waarden van de target (in eurocenten)
#'    y_hat  integer    de voorspelling van de actuele waarde (in eurocenten)
#'    w      double     het gewicht van de voorspelling (typisch een waarde tussen de 1/365 en 1)
#' returnwaarde: de (gewogen) R-squared
#'    
#' Opmerking 1: Dit hele project wordt uitgevoerd met als doel een model te vinden dat o.a. zo goed mogelijk de
#'              individuele kosten voorspelt van personen in de OT2020 data set VOORDAT er een herweging heeft
#'              plaatsgevonden. Het gewicht (gew) in een rij geeft aan voor welke fractie van het jaar de betreffende
#'              situatie geldig was. In voorgaande vereveningsonderzoeken was het de usance om de de target te delen 
#'              door dat gewicht en daarmee dus eigenlijk af te stappen van het voorspellen van de werkelijke kosten en 
#'              in plaats daarvan te voorspellen wat de kosten zouden zijn geweest als de verzekerde het hele jaar had
#'              geleefd (in geval geboorte en overlijden) en het hele jaar bij dezelfde verzekeraar zou zijn geweest
#'              (in het geval van overstappen) en de rest van het jaar dezelfde gemiddelde kosten per dag gemaakt zou 
#'              hebben. Omdat in de praktijk van het verevenen aan het einde van het jaar een
#'              nacalculatie plaatsvindt op basis van (in ieder geval) het aantal dagen van het jaar dat een persoon is
#'              ingeschreven geweest bij een verzekeraar kunnen we zonder bezwaar deze conventie volgen. Dit maakt de
#'              voorspelling ook iets anders omdat we dan dus geen rekening hoeven te houden met bijvoorbeeld de 
#'              kans op overlijden. Het is wel zo dat we door deze operatie extra data hebben aangemaakt. Als iemand
#'              bijvoorbeeld halverwege het jaar overstapt naar een andere verzekering dan wordt voor beide records de
#'              target met 2 vermenigvuldigd en hebben we een extra verzekeringsjaar gecre?erd. We MOGEN hier rekening
#'              mee houden op het moment dat de we de modellen trainen, we MOETEN hier rekening mee houden op het als we
#'              de score berekenen met behulp van een metric. De error op een voorspelling van een actuele targetwaarde
#'              die is vermenigvuldigd moet navenant minder meetellen. Overigens zijn de weging (naar rato van de 
#'              daadwerkelijke inschrijfduur) en de HERweging (waarbij gewichten worden aangepast adhv omstandigheden
#'              die veranderden tussen 2016 en 2020) twee verschillende zaken!
R2 <- function(y, y_hat, w){
  stopifnot(length(y) == length(y_hat))
  stopifnot(length(y) == length(w))
  if(length(unique(y)) < length(unique(y_hat))) cat("Warning: are you sure you didn't switch y and y_hat?\n")
  y_mean     <- weighted.mean(y, w)
  1 - (sum((y - y_hat)^2 * w) / sum((y - y_mean)^2 * w))
}

