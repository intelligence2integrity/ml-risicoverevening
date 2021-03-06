#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.<br/>
#' The accompanying reports were published under the names <br/>
#' "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.<br/>
#' "Minimaliseren van bandbreedte in de risicoverevening" on December 1st 2020.<br/>
#' The code for "Onderzoek Machine Learning in de Risicoverevening" (OML) was published on GitHub on June 10th 2020.<br/>
#' The code for "Minimaliseren van bandbreedte in de risicoverevening" (MB) was published on GitHub on December 31st 2020.<br/>
#' The copyright of this work is owned by the Ministry of Health, Welfare and Sport in the Netherlands.<br/>
#' Enquiries about the use of this code (e.g. for non-commercial purposes) are encouraged.<br/>

#' For questions contact:<br/>

#' i2i B.V.<br/>
#' info@i2i.eu<br/>

# Onderzoek Machine Learning in de Risicoverevening
In opdracht van het ministerie van VWS is onderzocht in hoeverre machine learning algoritmes het huidige systeem van
risicoverevening in de Nederlandse gezondheidszorg kunnen verbeteren.
In deze repository is de code terug te vinden die is gebruikt om de modellen te trainen. De gebruikte data is echter
niet openbaar dus de resultaten zelf zijn hiermee niet te reproduceren. De voor dit onderzoekte gebruikte data bevat
geen werkelijke BSN's, enkel door ZorgTTP gepseudonimiseerde BSN's.

# R2-score van de verschillende modellen op de test set
| Model | R2 | R2 van OLS model op dezelfde data | Gebruikt meer data dan OT | Verschil met gelijke data OLS | Verschil met OT-data OLS|
| ------------------ | ---- | ---- | --- | ---- | ---- |
| decision_tree_v026 | 35,0 | 35,1 | Nee | -0,1 | -0,1 |
| nn_v005            | 36,3 | 35,1 | Nee | 1,2  | 1,2  |
| nn_v010            | 38,2 | 36,2 | Ja  | 2,0  | 3,1  |
| RF_v002            | 36,3 | 35,1 | Nee | 1,2  | 1,2  |
| RF_v003            | 36,3 | 35,1 | Ja  | 1,2  | 1,2  |
| segmented_v005     | 35,3 | 35,1 | Ja  | 0,2  | 0,2  |
| XGB_only_OT_v001   | 36,3 | 35,1 | Nee | 1,2  | 1,2  |
| XGB_v015           | 38,5 | 36,1 | Ja  | 2,4  | 3,4  |
| OLS_v007           | 35,1 | 35,1 | Nee | 0,0  | 0,0  |

# Minimaliseren van bandbreedte in de risicoverevening
In aansluiting op het hierboven genoemde onderzoek is in opdracht van het ministerie van VWS onderzocht in hoeverre 
de bandbreedte te verbeteren viel met als startpunt XGB_v015 en nn_v010, zonder acht te slaan op andere metrics zoals bijvoorbeeld de R2-score.

# bandbreedte-score van de verschillende modellen op de test set
| Model(-combinatie)          | bandbreedte (euro) |   R2 | Gebruikt meer data dan OT |
| ----------------------------| ------------------ | ---- |-------------------------- |
| OLS_v007                    |                313 | 35,1 |Nee                        |
| XGB_v015                    |                317 | 38,5 |Ja                         |
| XGB_v040                    |                251 | 35,4 |Ja                         |
| XGB_v015 + booster_xgb_v011 |                215 | 29,7 |Ja                         |
| nn_v010                     |                257 | 38,2 |Ja                         |
| nn_v010 + booster_nn_v011   |                281 | 33,9 |Ja                         |

# Beschrijving files
```
.
├── README.md
├── renv.lock           # Voor reproduceren dependencies met renv 
├── vws_rve_ml.Rproj    # Rstudio project file
└── src
    ├── config
    │   └── VWS_Gupta_i2i_ML_RVE_options_v003.R  # Globale R opties
    ├── models
    │   ├── VWS_Gupta_i2i_ML_decision_tree_v026.R                  # decision tree op OT-features
    │   ├── VWS_Gupta_i2i_ML_doe_cross_validation_v034.R           # uitvoeren 10-folds cv
    │   ├── VWS_Gupta_i2i_ML_doe_train_v003.R                      # trainen op gehele trainset
    │   ├── VWS_Gupta_i2i_ML_doe_train_v003.R                      # (MB) trainen op gehele trainset
    │   ├── VWS_Gupta_i2i_ML_nn_bandbreedte_booster_v011.R         # (MB) booster voor het minimaliseren van de bandbreedte
    │   ├── VWS_Gupta_i2i_ML_nn_v005.R                             # neuraal netwerk op OT-features
    │   ├── VWS_Gupta_i2i_ML_nn_v010.R                             # neuraal netwerk met extra features
    │   ├── VWS_Gupta_i2i_ML_ols_met_nn_v010_feats_v002.R          # ols met features van nn_v010
    │   ├── VWS_Gupta_i2i_ML_ols_met_RF_v003_feats_v003.R          # ols met features van RF_v003
    │   ├── VWS_Gupta_i2i_ML_ols_met_segmented_v005_feats_v003.R   # ols met features van segmented_v005
    │   ├── VWS_Gupta_i2i_ML_ols_met_xgb_v015_feats_v003.R         # ols met features van xgb_v015
    │   ├── VWS_Gupta_i2i_ML_ols_v007.R                            # R-implementatie van huidige ols-model
    │   ├── VWS_Gupta_i2i_ML_predict_v002.R                        # getraind model testen op testset
    │   ├── VWS_Gupta_i2i_ML_predict_v004.R                        # (MB) getraind model testen op testset
    │   ├── VWS_Gupta_i2i_ML_RF_v002.R                             # random forest op OT-features
    │   ├── VWS_Gupta_i2i_ML_RF_v003.R                             # random forest met extra features
    │   ├── VWS_Gupta_i2i_ML_segmented_v005.R                      # gesegmenteerde regressie
    │   ├── VWS_Gupta_i2i_ML_xgb_bandbreedte_booster_v011.R        # (MB) booster voor het minimaliseren van de bandbreedte
    │   ├── VWS_Gupta_i2i_ML_XGB_only_OT_v001.R                    # xgboost op OT-features
    │   ├── VWS_Gupta_i2i_ML_XGB_v015.R                            # xgboost met extra features
    │   └── VWS_Gupta_i2i_ML_XGB_v040.R                            # (MB) xgboost geoptimaliseerd voor bandbreedte
    ├── R
    │   ├── VWS_Gupta_i2i_ML_calc_permutation_feature_importance_v005.R    # berekenen perm. feat. imp.
    │   ├── VWS_Gupta_i2i_ML_data_preparatie_v011.R                        # feature engineering
    │   ├── VWS_Gupta_i2i_ML_data_preparatie_v012.R                        # (MB) feature engineering
    │   ├── VWS_Gupta_i2i_ML_doe_verstelbare_hyperopt_cv_v008.R            # (MB) uitvoeren 10-folds cv (batch mode)
    │   ├── VWS_Gupta_i2i_ML_RVE_calc_scores_v011.R                        # bepalen scores op cv
    │   ├── VWS_Gupta_i2i_ML_RVE_doe_data_preparatie_v004.R                # uitvoeren feature engineering
    │   ├── VWS_Gupta_i2i_ML_RVE_get_feature_importance_tree_models_v001.R # feat. imp. voor dt, rf en xgb
    │   └── VWS_Gupta_i2i_ML_segmented_normbedragen_v002.R                 # normbedragen uit segm. regr.
    └── utils
        ├── VWS_Gupta_i2i_ML_utils_v003.R                                  # verschillende utility functies
        └── VWS_Gupta_i2i_ML_utils_v008.R                                  # (MB) verschillende utility functies
```
De namen van de files in directory src/models komen (na de prefix VWS_Gupta_i2i_ML_) overeen met de namen van de
modellen in de tabel met testscores.

# Disclaimer

Op dit werk berust auteursrecht (zie boven). i2i B.V. en de opdrachtgever (het Ministerie van VWS) geven geen garanties op beschikbaarheid, actualiteit en continuïteit van de broncode. i2i B.V. en de opdrachtgever zijn niet verantwoordelijk voor eventuele (financiële) gevolgen van het directe of indirecte gebruik van de broncode. i2i B.V. en de opdrachtgever aanvaarden geen aansprakelijkheid voor gevolgen door onjuistheden in de broncode. 
