#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.

#' For questions contact:

#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma:   VWS_Gupta_i2i_data_preparatie_vXXX.R
#' Doel:        data preparatie van train en test sets van OT2020
#' Auteurs:     Jules van Ligtenberg
#'              Diederik Perdok
#' Opmerking 1: Deze data preparatie moet zowel voor de test set als de train set gedraaid worden.
#' Opmerking 2: De operaties, uitgevoerd in dit programma, op de verschillende rijen in het bestand moeten onafhankelijk
#'              zijn van de inhoud van andere rijen. Dit is belangrijk omdat dit programma gedraaid wordt voor het
#'              opdelen van de data in train en test folds in de cross validation. Dit ter voorkoming van leakage.
#' Opmerking 3: In de splitsing van de train en de test set is reeds rekening gehouden met het feit dat er soms meerdere 
#'              rijen zijn met dezelfde 'Pseudoniem BSN'. Deze zijn bij elkaar OF in de train OF in de test set
#'              geplaatst. Op die manier wordt voorkomen dat er leakage optreedt. Op dezelfde manier moet hier tijdens
#'              het ontwerpen van cross validations rekening mee gehouden worden.
#' Opmerking 4: Dit hele project wordt uitgevoerd met als doel een model te vinden dat o.a. zo goed mogelijk de
#'              individuele kosten voorspelt van personen in de OT2020 data set VOORDAT er een herweging heeft
#'              plaatsgevonden. Het gewicht (gew) in een rij geeft aan voor welke fractie van het jaar de betreffende
#'              situatie geldig was. In voorgaande vereveningsonderzoeken was het de usance om de de target te delen 
#'              door dat gewicht en daarmee dus eigenlijk af te stappen van het voorspellen van de werkelijke kosten en 
#'              in plaats daarvan te voorspellen wat de kosten zouden zijn geweest als de verzekerde het hele jaar had
#'              geleefd (in geval geboorte en overlijden) en het hele jaar bij dezelfde verzekeraar zou zijn geweest
#'              (in het geval van overstappen). Omdat in de praktijk van het verevenen aan het einde van het jaar een
#'              nacalculatie plaatsvindt op basis van (in ieder geval) het aantal dagen van het jaar dat een persoon is
#'              ingeschreven geweest bij een verzekeraar kunnen we zonder bezwaar deze conventie volgen. Dit maakt de
#'              voorspelling ook iets makkelijker omdat we dan dus geen rekening hoeven te houden met bijvoorbeeld de 
#'              kans op overlijden. Het is wel zo dat we door deze operatie extra data hebben aangemaakt. Als iemand
#'              bijvoorbeeld halverwege het jaar overstapt naar een andere verzekering dan wordt voor beide records de
#'              target met 2 vermenigvuldigd en hebben we een extra verzekeringsjaar gecreëerd. We MOGEN hier rekening
#'              mee houden op het moment dat de we de modellen trainen, we MOETEN hier rekening mee houden op het als we
#'              de score berekenen met behulp van een metric. De error op een voorspelling van een actuele targetwaarde
#'              die is vermenigvuldigd moet navenant minder meetellen.
#'              
########################################################################################################################


#' preconditie: True
#' parameters: 
#'   naam       functie
#'   root_path  is de directory waar alles staat
#'   train      geeft aan of we de train set inlezen en aanpassen of de test set
#'   debug      geeft aan of we in debug mode gaan werken

prepare <- function(root_path, train=T, debug=F){
  
  version <- 11
  
  data_path <- paste0(root_path, 'data/')

  library(data.table)
  library(stringr)
  
  # settings
  source(paste0(root_path, 'src/config/VWS_Gupta_i2i_ML_RVE_options_v003.R'))

  ########################
  # Verwerken OT-dataset #
  ########################
  cat('Verwerken OT-set\n')
  if(train){
    file_name_data_set <- 'train'
  }else{
    file_name_data_set <- 'test'
  }
  if(debug){
    dt <- fread(paste0(data_path, file_name_data_set, '.csv'), nrows=99999)
    debug_string <- '_debug'
  }else{
    dt <- fread(paste0(data_path, file_name_data_set, '.csv'))
    debug_string <- ''
  }
  
  # leg aantal recs vast
  nrow_dt <- nrow(dt)
  
  # leg volgorde vast
  dt$ori_order <- 1:nrow(dt)

  names(dt)[names(dt) == 'Pseudoniem BSN'] <- 'Pseudoniem_BSN'
  # belangrijkste reden voor deze wijziging: moeilijker maken om per ongeluk de verkeerde versie te gebruiken
  for(i in 1:9){
    cur_name <- paste0('fkg', i)  
    new_name <- paste0('fkg0', i)
    names(dt)[which(names(dt) == cur_name)]       <- new_name
  }
  
  # aanmaken target en gewicht
  dt$target <- dt$verv + dt$huis + dt$parm + dt$farm + dt$vlos + dt$kraam + dt$hulp + dt$tand + dt$keten + dt$varb + 
               dt$diag + dt$grz  + dt$zg   + dt$vv   + dt$elv
  
  dt$gew <- dt$induur/365                # gewicht (zie ook opmerkingen hier boven)
  
  dt$target <- round(dt$target / dt$gew) 

  
  dt$fkg00 <- (dt$fkg01 + dt$fkg02 + dt$fkg03 + dt$fkg04 + dt$fkg05 + dt$fkg06 + dt$fkg07 + dt$fkg08 + dt$fkg09 +
               dt$fkg10 + dt$fkg11 + dt$fkg12 + dt$fkg13 + dt$fkg14 + dt$fkg15 + dt$fkg16 + dt$fkg17 + dt$fkg18 +
               dt$fkg19 + dt$fkg20 + dt$fkg21 + dt$fkg22 + dt$fkg23 + dt$fkg24 + dt$fkg25 + dt$fkg26 + dt$fkg27 +
               dt$fkg28 + dt$fkg29 + dt$fkg30 + dt$fkg31 + dt$fkg32 + dt$fkg33 + dt$fkg34 + dt$fkg35 + dt$fkg36 +
               dt$fkg37) == 0
  dt$fkg00 <- as.integer(dt$fkg00)
  
  dt$verv    <- NULL
  dt$huis    <- NULL
  dt$parm    <- NULL
  dt$farm    <- NULL
  dt$vlos    <- NULL
  dt$kraam   <- NULL
  dt$hulp    <- NULL
  dt$tand    <- NULL
  dt$keten   <- NULL
  dt$varb    <- NULL 
  dt$vast    <- NULL
  dt$diag    <- NULL
  dt$grz     <- NULL
  dt$zg      <- NULL
  dt$vv      <- NULL
  dt$elv     <- NULL 
  dt$hergew  <- NULL # vanaf versie 5 eruit
  dt$rdragnw <- NULL # vanaf versie 5 eruit
  dt$laagt_3 <- NULL # vanaf versie 5 eruit
  dt$hoogt_3 <- NULL # vanaf versie 5 eruit

  dt$variabel_op <- NULL # dit is de target die gebruikt wordt bij de herberekening
  
  #########################################################
  # toevoegen continue leeftijd, o.b.v eerste aanlevering #
  #########################################################
  cat('Toevoegen continue leeftijd\n')
  setkey(dt, 'Pseudoniem_BSN')
  for(jaar in 2017:2016){
    # tijdens de trainperiode mogen we alleen de leeftijden zien van diegenen die in de train set zitten..
    leeftijd_data <- fread(paste0(data_path, 'GUPTA_RVE_ZINL_PER_', jaar, '_1.csv'), nrows = if (debug) 1e5 else Inf)
    names(leeftijd_data)[names(leeftijd_data) == 'Pseudoniem BSN'] <- 'Pseudoniem_BSN'
    cat(jaar, '\n')
    
    leeftijd_data[, setdiff(names(leeftijd_data), c('Pseudoniem_BSN', 'Leeftijd')) := NULL]
    
    #' het komt sporadisch voor dat er dubbele BSN's in het bestand voorkomen en nog minder vaak dat deze verschillende
    #' leeftijden hebben. Om toch iets eenduidigs te krijgen aggregeren we eerst een en ander
    leeftijd_data <- leeftijd_data[, .(Leeftijd = mean(Leeftijd)), by = Pseudoniem_BSN]
    setkey(leeftijd_data, 'Pseudoniem_BSN')
    dt <- merge(dt, leeftijd_data, all.x = T)
    names(dt)[names(dt) == 'Leeftijd'] <- paste0('Leeftijd_', jaar)

    stopifnot(nrow(dt) == nrow_dt)
  }
  diff <- dt$Leeftijd_2017 - dt$Leeftijd_2016
  print(table(diff))
  # maak er de leeftijd in 2017 van
  dt$Leeftijd_2016 <- dt$Leeftijd_2016 + 1
  dt$leeftijd <- rowMeans(dt[,c('Leeftijd_2016', 'Leeftijd_2017')], na.rm=T)
  dt$Leeftijd_2016 <- NULL
  dt$Leeftijd_2017 <- NULL
  
  #########################################################
  # toevoegen continue leeftijd, o.b.v heraanlevering     #
  #########################################################
  cat('Toevoegen continue leeftijd2\n')
  for(jaar in 2017:2016){
    for(soort_bestand in c('_PER_', '_PKB_')){
      # tijdens de trainperiode mogen we alleen de leeftijden zien van diegenen die in de train set zitten..
      bestandsnaam <- paste0(data_path, '/bron_data/heraanlevering_per_pkb/', 'GUPTA_RVE_ZINL', soort_bestand, jaar,
                             if (soort_bestand == '_PER_') '_2' else '_1', '.csv')
      cat(paste0('Bezig met ', bestandsnaam, '\n'))
      leeftijd_data <- fread(bestandsnaam, nrows = if (debug) 99999 else Inf)
      names(leeftijd_data)[names(leeftijd_data) == 'Pseudoniem BSN'] <- 'Pseudoniem_BSN'

      # geboortedatum (eigenlijk geboortemaand) naar leeftijd in juni vertalen. Daarmee komen de leeftijden van de
      # 'PER'-bestanden overeen met de 'PER' bestanden uit de vorige aanlevering (hierin werd de leeftijd in plaats van
      # de geboortemaand vermeld)
      leeftijd_data[, geb_maand := as.integer(substr(as.character(geb_datum), 5, 6))]
      leeftijd_data[, geb_jaar := as.integer(substr(as.character(geb_datum), 1, 4))]
      leeftijd_data[, Leeftijd := 2017 - geb_jaar]
      leeftijd_data[geb_maand >= 7, Leeftijd := Leeftijd - 1]
      leeftijd_data[, setdiff(names(leeftijd_data), c('Pseudoniem_BSN', 'Leeftijd')) := NULL]

      #' het komt sporadisch voor dat er dubbele BSN's in het bestand voorkomen en nog minder vaak dat deze verschillende
      #' leeftijden hebben. Om toch iets eenduidigs te krijgen aggregeren we eerst een en ander
      leeftijd_data <- leeftijd_data[, .(Leeftijd = mean(Leeftijd)), by = Pseudoniem_BSN]
      setkey(leeftijd_data, 'Pseudoniem_BSN')
      dt <- merge(dt, leeftijd_data, all.x = T)
      names(dt)[names(dt) == 'Leeftijd'] <- paste0('Leeftijd', soort_bestand, jaar)

      stopifnot(nrow(dt) == nrow_dt)
    }
  }
  dt[, leeft_gelijk := (Leeftijd_PER_2016 == Leeftijd_PER_2017) & (Leeftijd_PER_2017 == Leeftijd_PKB_2016)
                       & (Leeftijd_PKB_2016 == Leeftijd_PKB_2017)]
  print(paste('In', nrow(dt), 'records zijn er', nrow(dt[leeft_gelijk == F]) + nrow(dt[is.na(leeft_gelijk)]),
              'met een verschillende leeftijd'))

  # Neem het gemiddelde van de 4 leeftijden
  dt$leeftijd2 <- rowMeans(dt[,c('Leeftijd_PER_2016', 'Leeftijd_PER_2017', 'Leeftijd_PKB_2016', 'Leeftijd_PKB_2017')],
                           na.rm=T)
  dt$Leeftijd_PER_2016 <- NULL
  dt$Leeftijd_PER_2017 <- NULL
  dt$Leeftijd_PKB_2016 <- NULL
  dt$Leeftijd_PKB_2017 <- NULL
  dt$leeft_gelijk      <- NULL

  cat('Aantal records zonder continue leeftijd:', sum(is.na(dt$leeftijd2)), '\n')
  
  ##########################################################
  # Toevoegen features van dx-groepen o.b.v. ruwe brondata #
  ##########################################################
  cat('Bepalen features DX-groepen\n')
  # We volgen grotendeels de methode van officiële OLS-model o.b.v. declaraties uit 2016 (bron: "Uitvoeringstabellen
  # somatische DKGs verzekerdenraming 2019" + apart aangeleverde lijst met uit te sluiten consult zorgproducten) met een
  # paar verschillen:
  # - Geen leeftijdsrestricties
  # - Geen interacties tussen verschillende dxgroepen ("als in groep a maar niet in b dan wordt het groep c").
  # - Geen verwerkingsstappen die gebaseerd zijn op data uit eerder jaren.
  
  # Ophalen relevante zorgproducten (selectie door ZIN). Ontbrekende voorloopnullen vullen we aan. Er zijn 2 types:
  # - Consulten: Declaraties hiervan leiden nooit tot een dx-groep, ongeacht de diagnose
  # - Producten die rechtstreeks leiden tot indeling in een dx-groep, ongeacht de diagnose
  # Er is 1 zorgproduct dat in beide series voor komt (het leidt tot een dxgroep maar is ook een uit te sluiten
  # consult). Er is voor gekozen om deze niet uit te sluiten en wel tot de dxgroep te laten leiden.
  dt_zorgproduct <- fread(paste0(data_path, 'bron_data/indeling_brondata_in_ot_kenmerken/MAN_ZPRD_DKG.csv'),
                          select = c('ZPRD_CODE', 'DX', 'CONSULT'),
                          col.names = c('zp_code', 'dxgroep', 'consult'))
  dt_zorgproduct[, zp_code := str_pad(zp_code, width = 9, pad = '0')]
  dt_zorgproduct[zp_code == '182199037', consult := 0] # 182199037 is de uitzondering
  dt_zp2dx <- dt_zorgproduct[consult == 0, .(zp_code, dxgroep)]
  dt_zp_consult <- dt_zorgproduct[consult == 1, zp_code] 
  dt_zorgproduct <- NULL
  
  # Ophalen unieke diagnoses en zorgproducten per persoon. Ontbrekende voorloopnullen moeten aangevuld worden, en het
  # het bronbestand heeft quotes rondom elke waarde, waarbinnen witruimte voor kan komen. We maken alle diagnosecodes
  # simpelweg 4 cijfers lang, ook bij specialismes die diagnosecodes van 2 of 3 tekens (horen te) hebben. Consulten
  # gooien we meteen weg.
  dt_dbc <- fread(paste0(data_path, 'bron_data/deel_2/GUPTA_RVE_ZINL_DBC_2016_1.csv'),
                  select = c('Pseudoniem BSN', 'DIAG_code', 'SPEC_code', 'ZP_code'),
                  col.names = c('Pseudoniem_BSN', 'diagnose_code', 'specialisme_code', 'zp_code'),
                  nrows = if (debug) 99999 else Inf) # declaraties
  dt_dbc[, Pseudoniem_BSN := str_trim(Pseudoniem_BSN, side = 'both')]
  dt_dbc[, diagnose_code := str_pad(str_trim(diagnose_code, side = 'both'), width = 4, pad = '0')]
  dt_dbc[, zp_code := str_pad(str_trim(zp_code, side = 'both'), width = 9, pad = '0')]
  dt_dbc <- dt_dbc[!(zp_code %in% dt_zp_consult)]
  dt_dbc_diag <- unique(dt_dbc[, .(Pseudoniem_BSN, diagnose_code, specialisme_code)])
  dt_dbc_zp <- unique(dt_dbc[, .(Pseudoniem_BSN, zp_code)])
  
  # Ophalen vertaling diagnose naar DX-groep (vertaling door ZIN)
  dt_diag2dx <- fread(paste0(data_path, 'bron_data/indeling_brondata_in_ot_kenmerken/MAN_DIAG_DKG_V2.csv'),
                      select = c('SPEC', 'DIAG', 'DX'),
                      col.names = c('specialisme_code', 'diagnose_code', 'dxgroep'))
  dt_diag2dx[, diagnose_code := str_pad(diagnose_code, width = 4, pad = '0')]
  
  # Bepalen DX-groepen o.b.v. diagnoses en zorgproducten
  dt_dx_obv_diag <- merge(dt_dbc_diag, dt_diag2dx, by = c('specialisme_code', 'diagnose_code'))
  dt_dx_obv_diag <- unique(dt_dx_obv_diag[, .(Pseudoniem_BSN, dxgroep)])
  dt_dx_obv_zp <- merge(dt_dbc_zp, dt_zp2dx, by = 'zp_code')
  dt_dx_obv_zp <- unique(dt_dx_obv_zp[, .(Pseudoniem_BSN, dxgroep)])
  
  # Samenvoegen DX-groepen uit beide bronnen. DX-groep 1750 geldt alleen wanneer een persoon zowel o.b.v. diagnoses als
  # o.b.v. zorgproducten in de DX-groep valt.
  bsn_1750 <- intersect(dt_dx_obv_diag[dxgroep == 1750, Pseudoniem_BSN],
                        dt_dx_obv_zp[dxgroep == 1750, Pseudoniem_BSN])
  dt_dx_obv_diag <- dt_dx_obv_diag[(dxgroep == 1750) & (Pseudoniem_BSN %in% bsn_1750) | (dxgroep != 1750)]
  dt_dx_obv_zp <- dt_dx_obv_zp[(dxgroep == 1750) & (Pseudoniem_BSN %in% bsn_1750) | (dxgroep != 1750)]
  
  dt_dx <- rbind(dt_dx_obv_diag, dt_dx_obv_zp)
  dt_dx <- unique(dt_dx)
  dt_dx_obv_diag <- NULL
  dt_dx_obv_zp <- NULL
  
  # Tabel "pivotten" naar wide-format
  dt_dx[, dxgroep := paste0('dx_groep_', dxgroep)]
  dx_cols <- dt_dx[, unique(dxgroep)]
  dt_dx <- dcast(dt_dx, Pseudoniem_BSN ~ dxgroep, fun.aggregate = length, fill = 0, value.var = 'dxgroep')
  
  # mergen met rest van de train set
  dt <- merge(dt, dt_dx, by = 'Pseudoniem_BSN', all.x = T)
  for (i in dx_cols)
    dt[is.na(get(i)), (i) := 0]
  
  ###############################################################################
  # Toevoegen features geschat aantal verpleegdagen en operatieve verrichtingen #
  ###############################################################################
  cat('Bepalen features aantal verpleegdagen en operatieve verrichtingen\n')
  # Bron: Gemiddelde zorgprofielen uit bestand "20170101 Zorgproduct Profielen v20160701.csv", uit DBC-pakket RZ17B,
  # gepubliceerd op 17-11-2016. I.c.m. declaraties van zorgproducten (met consulten reeds uitgesloten) bepalen we
  # hiermee de verwachtingswaarde van het aantal verpleegdagen (zpk = 3) en operatieve verrichtingen (zpk = 5) per
  # persoon, waarbij we individuele declaraties van zorgproducten als onafhankelijk beschouwen; met andere woorden: we
  # tellen het gemiddelde aantal zorgactiviteiten per zorgproduct gewoon op.
  dt_profiel <- fread(paste0(data_path, 'bron_data/indeling_brondata_in_ot_kenmerken',
                             '/20170101 Zorgproduct Profielen v20160701.csv'),
                      select = c('zorgproduct_code', 'zorgactiviteit_code', 'zpk_code', 'percentage_voorkomen_RZ16',
                                 'gemiddeld_aantal_geregistreerd_RZ16'),
                      col.names = c('zp_code', 'za_code', 'zpk_code', 'perc_voorkomen', 'gem_aantal'),
                      dec = ',')
  dt_profiel[, zp_code := str_pad(str_trim(zp_code, side = 'both'), width = 9, pad = '0')]
  dt_profiel <- dt_profiel[(zpk_code %in% c(3, 5)) & (gem_aantal != '') & (perc_voorkomen != '')]
  dt_profiel <- dt_profiel[, .(aantal_tot = sum(perc_voorkomen * gem_aantal)), by = .(zp_code, zpk_code)]
  dt_aant_vplg <- merge(dt_dbc, dt_profiel, by = 'zp_code') # 0.39% vd declaraties heeft een zp niet in dt_profiel
  dt_aant_vplg <- dt_aant_vplg[, .(verwacht_aantal = sum(aantal_tot)), by = .(Pseudoniem_BSN, zpk_code)]
  dt_aant_vplg <- dcast(dt_aant_vplg, Pseudoniem_BSN ~ zpk_code, fill = 0, value.var = 'verwacht_aantal')
  names(dt_aant_vplg) <- c('Pseudoniem_BSN', 'n_verpleeg', 'n_operatief')
  
  # mergen met rest van de train set
  dt <- merge(dt, dt_aant_vplg, by = 'Pseudoniem_BSN', all.x = T)
  dt[is.na(n_verpleeg), n_verpleeg := 0]
  dt[is.na(n_operatief), n_operatief := 0]
  dt_aant_vplg <- NULL
  
  ###################################
  # Toevoegen continue FKG-features #
  ###################################
  # "Continu" wil zeggen aantal DDD's voor farmacie-FKG's (extramuraal) en aantal eenheden voor add-on-FKG's
  cat('Bepalen continue fkg-features\n')
  
  dt_farm_to_fkg <- fread(paste0(data_path, 'bron_data/indeling_brondata_in_ot_kenmerken/MAN_FAR_FKG_2.csv'),
                       select = c('ATC_Code', 'FKG', 'FKG_ID'),
                       col.names = c('atc_code', 'fkg_naam', 'fkg_code'))

  list_dt_farm_maand <- vector('list', 12)
  for (maand_nr in 1:12){
    cat(paste('Apotheekdeclaraties maand', maand_nr, '\n'))
    dt_farm_maand <- fread(paste0(data_path, 'bron_data/deel_3/GUPTA_RVE_ZINL_FAR_2016_',
                                  str_pad(maand_nr, width = 2, pad = '0'), '_1.csv'),
                           select = c('Pseudoniem BSN', 'ATC_code', 'HPK_ddd', 'HOEVEEL', 'DCind'),
                           col.names = c('Pseudoniem_BSN', 'atc_code', 'ddd_per_eenheid', 'aantal_eenheden', 'dc_ind'),
                           nrows = if (debug) 99999 else Inf)
    dt_farm_maand <- dt_farm_maand[dc_ind == 'D']
    dt_farm_maand <- merge(dt_farm_maand, dt_farm_to_fkg, by = 'atc_code') # filter relevante atc
    cat(paste('Fractie relevante declaraties zonder DDD:',
              nrow(dt_farm_maand[ddd_per_eenheid == 0]) / nrow(dt_farm_maand), '\n'))
    
    dt_farm_maand <- dt_farm_maand[, .(aantal_ddd = sum((aantal_eenheden / 1000) * (ddd_per_eenheid / 1000))),
                                   by = c('Pseudoniem_BSN', 'atc_code')]
    names(dt_farm_maand)[3] <- paste0('aantal_ddd_', maand_nr)
    setkey(dt_farm_maand, Pseudoniem_BSN, atc_code)
    
    list_dt_farm_maand[[maand_nr]] <- dt_farm_maand
  }
  
  cat(paste('Apotheekdeclaraties maanden samenvoegen\n'))
  dt_farm <- merge(list_dt_farm_maand[[1]], list_dt_farm_maand[[2]], all = T)
  dt_farm[is.na(aantal_ddd_1), aantal_ddd_1 := 0]
  dt_farm[is.na(aantal_ddd_2), aantal_ddd_2 := 0]
  dt_farm[, aantal_ddd := aantal_ddd_1 + aantal_ddd_2]
  for (maand_nr in 3:12){
    aantal_ddd_var_name <- paste0('aantal_ddd_', maand_nr)
    dt_farm <- merge(dt_farm, list_dt_farm_maand[[maand_nr]], all = T)
    dt_farm[is.na(get(aantal_ddd_var_name)), (aantal_ddd_var_name) := 0]
    dt_farm[is.na(aantal_ddd), aantal_ddd := 0]
    dt_farm[, aantal_ddd := aantal_ddd + get(aantal_ddd_var_name)]
  }
  dt_farm <- dt_farm[, .(Pseudoniem_BSN, atc_code, aantal_ddd)]
  
  cat("Vormen FKG's o.b.v. extramurale farmacie\n")
  dt_fkg_obv_farm <- merge(dt_farm, dt_farm_to_fkg, by = 'atc_code')
  dt_fkg_obv_farm <- dt_fkg_obv_farm[, .(aantal_ddd = sum(aantal_ddd)), by = c('Pseudoniem_BSN', 'fkg_code')]
  dt_farm_maand <- NULL
  dt_farm <- NULL
  
  cat('Addon declaraties\n')
  dt_addon_to_fkg <- fread(paste0(data_path, 'bron_data/indeling_brondata_in_ot_kenmerken/MAN_ADDON_FKG.csv'),
                        select = c('Declaratie-code', 'FKG'),
                        col.names = c('decl_code', 'fkg_code'))
  dt_addon_to_fkg <- dt_addon_to_fkg[fkg_code != 0]
  
  dt_addon <- fread(paste0(data_path, 'bron_data/deel_1/GUPTA_RVE_ZINL_1/GUPTA_RVE_ZINL_ADDON_2016_1.csv'),
                    select = c('Pseudoniem BSN', 'decl_code_add', 'hoeveel', 'DCind'),
                    col.names = c('Pseudoniem_BSN', 'decl_code', 'aantal_eenheden', 'dc_ind'),
                    nrows = if (debug) 99999 else Inf)
  
  dt_addon <- dt_addon[dc_ind == 'D']
  dt_addon <- merge(dt_addon, dt_addon_to_fkg, by = 'decl_code') # filter relevante declaraties
  dt_addon <- dt_addon[, .(aantal_eenheden = sum(aantal_eenheden)), by = c('Pseudoniem_BSN', 'decl_code')]
  
  cat("Vormen FKG's o.b.v. add-ons\n")
  dt_fkg_obv_addon <- merge(dt_addon, dt_addon_to_fkg, by = 'decl_code')
  dt_fkg_obv_addon <- dt_fkg_obv_addon[, .(aantal_eenheden = sum(aantal_eenheden)),
                                       by = c('Pseudoniem_BSN', 'fkg_code')]
  dt_addon <- NULL
  
  cat("Samenvoegen en casten FKG's en mergen met rest van de dataset\n")
  dt_fkg <- merge(dt_fkg_obv_farm, dt_fkg_obv_addon, by = c('Pseudoniem_BSN', 'fkg_code'), all = T)
  cat(paste(nrow(dt_fkg[(!is.na(aantal_ddd)) & (!is.na(aantal_eenheden))]), 'persoon-fkgs bepaald o.b.v. van farma',
           'en addons gezamenlijk\n'))
  
  dt_fkg[is.na(aantal_ddd), aantal_ddd := 0]
  dt_fkg[is.na(aantal_eenheden), aantal_eenheden := 0]
  dt_fkg[, c("aantal", "aantal_ddd", "aantal_eenheden") := .(aantal_ddd + aantal_eenheden, NULL, NULL)]
  
  dt_fkg[, fkg_code := paste0('aantal_fkg_', str_pad(fkg_code, width = 2, pad = '0'))]
  fkg_cols <- unique(dt_fkg[, fkg_code])
  dt_fkg <- dcast(dt_fkg, Pseudoniem_BSN ~ fkg_code, fill = 0, value.var = 'aantal')
  
  dt <- merge(dt, dt_fkg, by = 'Pseudoniem_BSN', all.x = T)
  dt_fkg <- NULL
  for (i in fkg_cols)
    dt[is.na(get(i)), (i) := 0]
  
  ###############################################################################################
  # Toevoegen features aantal zorgproducten en aantal (verschillende) diagnoses en specialismes #
  ###############################################################################################
  # N.B. Consult-DBC's zijn op dit punt reeds uitgesloten en worden bewust niet meegeteld
  cat('Toevoegen features aantal zorgproducten en aantal (verschillende) diagnoses en specialismes\n')
  
  dt_dbc[, specialisme_diagnose_code := paste0(specialisme_code, '_', diagnose_code)]
  dt_aant_diag <- dt_dbc[, .(aantal_zp = .N, aantal_diag = uniqueN(specialisme_diagnose_code),
                             aantal_spec = uniqueN(specialisme_code)), by = 'Pseudoniem_BSN']
  dt <- merge(dt, dt_aant_diag, by = 'Pseudoniem_BSN', all.x = T)
  dt[is.na(aantal_zp), aantal_zp := 0]
  dt[is.na(aantal_diag), aantal_diag := 0]
  dt[is.na(aantal_spec), aantal_spec := 0]
  
  dt_dbc <- NULL
  
  ################
  # Wegschrijven #
  ################
  cat('Afronden en wegschrijven\n')
  # zet records in originele volgorde ivm cross validation
  dt <- dt[order(dt$ori_order), ]
  dt$ori_order <- NULL

  file_name_new_data_set <- paste0(data_path, file_name_data_set, '_set_v', str_pad(version, width = 3, pad = '0'),
                                   debug_string, '.csv')
  file_name_source <- paste0(root_path, 'src/R/VWS_Gupta_i2i_ML_data_preparatie_v',
                             str_pad(version, width = 3, pad = '0'), '.csv')
  fwrite(dt, file=file_name_new_data_set, quote=F, row.names=F)
  
  if(!debug) {
    Sys.chmod(file_name_new_data_set, mode = "0444", use_umask = T)
    Sys.chmod(file_name_source, mode = "0444", use_umask = T)
  }
  cat('done')
}

