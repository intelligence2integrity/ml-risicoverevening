#' This work was commissioned by the Ministry of Health, Welfare and Sport in the Netherlands.
#' The accompanying report was published under the name "Onderzoek Machine Learning in de Risicoverevening" on May 29th 2020.
#' This code was published on GitHub on June 10th 2020.
#' This work is protected by copyright.
#' For questions contact:
#' i2i B.V.
#' info@i2i.eu

########################################################################################################################
#' Programma: globale settings voor de R-scripts
#' Doel:      Dit is vooral om ervoor te zorgen dat de verschillende modellen niet toevallig goed draaien vanwege
#'            niet opgemerkte speciale settings. Daarnaast helpt dit om onze resultaten te reproduceren op systemen met
#'            andere instellingen. De meeste modellen zullen goed draaien onafhankelijk van deze instellingen. 
#' Auteurs:   Jules van Ligtenberg
#'            Diederik Perdok
########################################################################################################################

# settings
options(add.smooth=T)
options(askYesNo=T)
options(browserNLdisabled=F)
options(CBoundsCheck=F)
options(check.bounds=F)
options(citation.bibtex.max=1)
options(continue="+ ")
options(defaultPackages=c("datasets", "utils", "grDevices", "graphics", "stats", "methods"))
options(demo.ask="default")
options(deparse.cutoff=60)
options(digits=9)
options(example.ask='default')
options(echo=T)
options(encoding='native.enc')
options(expressions=5000)
options(keep.source=T)
options(na.action='na.omit')
options(OutDec='.')
options(PCRE_limit_recursion=NA)
options(PCRE_study=10)
options(PCRE_use_JIT=T)
options(scipen=9)
options(show.error.messages=T)
options(stringsAsFactors=F)                # maakt van strings niet automatisch factors bij het inlezen
options(timeout=60)
options(warn=2)                            # zorgt ervoor dat waarschuwingen onmiddellijk resulteren in een foutmelding
options(warnPartialMatchArgs=T)            
options(warnPartialMatchAttr=T)
options(warnPartialMatchDollar=T)
options(warning.length=8170)
options(width=120)



