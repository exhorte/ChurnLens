# Audit initial du dépôt

> Audit réalisé le 2026-10-26 sur le dépôt tel qu'il était **avant** toute modification
> (commit `1da61ba` « second update », branche `main`, remote `origin` = `exhorte/ChurnLens`).
> Environnement de vérification : Python 3.14.5, dépendances du `requirements.txt` déjà
> installées dans `venv/` (pandas 3.0.3, scikit-learn 1.8.0, streamlit 1.57.0, fastapi 0.136.3).

**Légende.** Chaque constat est marqué :

- **[V]** *vérifié* — observé en exécutant le code ou en lisant précisément le fichier cité ;
- **[H]** *hypothèse* — plausible mais non démontrée, avec le test qui permettrait de trancher.

Les scripts d'audit (non conservés dans le dépôt) ont été exécutés sur une **copie** du projet pour ne
rien écraser : réentraînement, chargement du modèle sauvegardé, appels API (`TestClient`),
exécution de chaque page Streamlit (`streamlit.testing.v1.AppTest`).

---

## 1. Architecture actuelle

```
04_code/
├── app.py                  Streamlit (704 lignes) : 5 pages démo + page « Your own data »
├── train_model.py          entraînement : LR + RF, sélection par CV, rapport, sérialisation
├── conftest.py             ajoute la racine au sys.path (aucun test à côté)
├── src/
│   ├── config.py           constantes (en grande partie inutilisées)
│   ├── data_loader.py      chargement CSV, garde-fou « données synthétiques », nettoyage
│   ├── feature_engineering.py   séparation X/y + wrapper historique
│   ├── pipeline.py         FeatureEngineer + ColumnTransformer + build_pipeline()
│   ├── explain.py          contributions exactes de la régression logistique
│   └── automl.py           entraînement « bring your own data » (LR + RF)
├── service/main.py         API FastAPI : /, /health, /model, /predict, /predict/batch
├── models/                 churn_pipeline.pkl (9 Ko) + metadata.json
├── reports/                model_comparison.csv, feature_importance.csv
├── data/                   WA_Fn-UseC_-_Telco_Customer_Churn.csv (7 043 lignes)
├── notebooks/              01_EDA.ipynb, 02_Model_Training.ipynb (aucune sortie enregistrée)
├── scripts/                generate_figures.py, generate_pdf.py
├── Dockerfile, .dockerignore, .streamlit/config.toml, requirements.txt
└── docs/                   vide
```

Flux : `CSV → clean_data → split 80/20 stratifié → Pipeline(FeatureEngineer → ColumnTransformer → classifieur)`
→ CV 5 plis sur le train pour choisir le modèle → ajustement → un seul `churn_pipeline.pkl` relu par
l'app Streamlit et l'API.

**Absents** [V] : `README.md`, `LICENSE`, `tests/`, `.github/workflows/`, `data/README.md`, toute
documentation du jeu de données. Le dépôt n'a **qu'un seul commit** et aucun remote `upstream`.

## 2. Fonctionnalités existantes

Pipeline scikit-learn unique entraînement/inférence · comparaison LR vs Random Forest · app Streamlit
(vue d'ensemble, insights, prédiction individuelle, détail du modèle, aperçu des données, entraînement
sur CSV utilisateur) · explication individuelle exacte pour la régression logistique · importance par
permutation · API FastAPI (prédiction unitaire et par lot) · Dockerfile de l'API.

## 3. Fonctionnalités réellement opérationnelles (vérifiées par exécution)

| Élément | Preuve |
|---|---|
| `train_model.py` s'exécute de bout en bout | [V] CV ROC-AUC LR 0,8508 ± 0,0118, RF 0,8453 ± 0,0087 ; test LR : ROC-AUC 0,8472, F1 0,6113 — **identiques** à `models/metadata.json` |
| Le pipeline sauvegardé donne les mêmes scores que l'entraînement | [V] rechargé sur le split recalculé : ROC-AUC 0,8472, AP 0,6653 |
| Robustesse du pipeline à l'ordre des colonnes, colonnes en trop, catégories inconnues | [V] ordre inversé → même probabilité ; catégorie `Contract="Three year"` → score produit |
| API : `/health`, `/model`, `/predict`, `/predict/batch` | [V] `TestClient` : 200 sur requêtes valides, 422 sur `Contract="Weekly"` |
| Les 5 pages démo Streamlit et le bouton « Predict » | [V] `AppTest` : 0 exception, 0 erreur |
| Entraînement « votre propre CSV » (`automl.py`) | [V] fonction appelée directement sur un DataFrame synthétique ; la page n'a pas été pilotée (upload de fichier) |
| Reproductibilité de `reports/model_comparison.csv` | **[V] partielle** : la ligne LR est identique ; la ligne Random Forest diffère légèrement (accuracy 0,7722 recalculée contre 0,7715 versionnée) → le rapport committé n'a pas été produit par la configuration actuelle (cause non identifiée) |

## 4. Fonctionnalités incomplètes ou fragiles

- Aucun test automatisé, aucune CI [V].
- Seuil fixé à 0,5 partout (`train_model.py`, `app.py:343`, `service/main.py:35`) et jamais analysé [V].
- Pas de modèle de référence (Dummy), ni arbre de décision, ni boosting ; 2 modèles seulement [V].
- Pas d'analyse des faux positifs / faux négatifs, pas de matrice de confusion dans l'app démo [V].
- Pas de simulateur économique, pas d'export CSV de prédictions [V].
- Explication individuelle **uniquement** si le modèle retenu est linéaire (`explain.supports_explanation`) ;
  si la sélection par CV retenait la forêt, l'app n'afficherait plus aucune explication [V par lecture].
- Les messages d'erreur du pipeline sont ceux de scikit-learn : colonne manquante →
  `columns are missing: {'TenureSegment', 'tenure'}` (expose un nom de variable *dérivée*) ;
  `SeniorCitizen="Yes"` → `Cannot use median strategy with non-numeric data` [V].
- Dockerfile fonctionnel à l'origine [V] (voir §10), mais image inutilement lourde.

## 5. Problèmes de qualité du code

| # | Fichier | Constat | Conséquence | Correction |
|---|---|---|---|---|
| Q1 | `src/config.py`, `train_model.py` | `config.py` n'est importé par personne (hors `__init__`) ; `train_model.py` redéfinit `RANDOM_STATE`, `CV_FOLDS`, chemins. `PRIMARY_METRIC='f1'` et `METRIC_THRESHOLD=0.55` contredisent le code (sélection sur ROC-AUC) [V] | Deux sources de vérité, configuration trompeuse | Une seule configuration (`configs/config.toml` + `config.py`) réellement utilisée |
| Q2 | `app.py`, `service/main.py`, `train_model.py` | `sys.path.insert(...)` pour importer `src` ; paquet nommé `src` (nom générique et fragile) [V] | Imports dépendants du répertoire courant | Vrai paquet `churnguard` (`pyproject.toml`) |
| Q3 | `app.py` | Monolithe de 704 lignes ; chemins relatifs (`'models/churn_pipeline.pkl'`) qui exigent d'être lancé depuis la racine ; listes de choix et bornes des formulaires écrites en dur [V] | Divergence possible avec le modèle (voir §9) | Pages séparées, chemins issus de la configuration |
| Q4 | `train_model.py`, `app.py`, `automl.py` | `except Exception` larges (3 occurrences) ; messages par `print` avec émojis [V] | Erreurs masquées, pas de journalisation exploitable | Exceptions explicites (`DataValidationError`…), `logging` |
| Q5 | `data_loader.validate_data` | Renvoie un booléen que `train_model.py` **ignore** [V] | Une validation qui échoue n'arrête rien | Validation qui lève une exception |
| Q6 | `scripts/generate_pdf.py` | Chiffres écrits en dur (« ROC-AUC 0,847 », « 43 % vs 3 % »…), récit « From a Broken Model to ROC-AUC 0.85 » issu du dépôt d'origine, mention d'une « performance gate » de CI qui n'existe pas ; `reportlab` absent de `requirements.txt` [V] | Métriques non régénérées, affirmations invérifiables | Supprimer (reste dans l'historique git) |
| Q7 | `scripts/generate_figures.py`, notebooks | Dépendent de l'ancienne API `src.*` ; écrit dans `screenshots/` qui n'existe pas ; notebooks sans aucune sortie enregistrée [V] | Cassés après toute réorganisation | Mettre à jour ou régénérer |
| Q8 | `src/__init__.py` | `__author__ = "Davide Infantino"` (auteur du dépôt d'origine) et description du dépôt source [V] | Attribution à traiter correctement (voir §8 G1) | Conserver l'attribution dans `NOTICE`, mettre à jour le paquet |

## 6. Problèmes de méthodologie ML

| # | Fichier | Constat | Conséquence | Correction |
|---|---|---|---|---|
| **M1** | `train_model.py:51-60` | `class_weight="balanced"` sur les deux modèles. **[V]** Probabilité moyenne prédite sur le jeu de test = **0,413** pour un taux de churn observé de **0,265** ; 42,7 % des clients sont signalés à 0,5 | Probabilités **mal calibrées** : toute espérance calculée à partir d'elles (nombre de départs évités, bénéfice) serait surestimée d'environ 55 % ; les bandes de risque 0,3/0,7 n'ont pas de sens probabiliste | Entraîner sans pondération, mesurer la calibration (Brier, probabilité moyenne vs taux observé), gérer le compromis précision/rappel **par le seuil** |
| **M2** | partout | Seuil de décision 0,5 en dur, aucune analyse de seuils [V] | Le seuil réel d'une campagne est un choix économique, pas 0,5 | Table de seuils calculée sur les prédictions hors-pli (OOF) du train ; seuil recommandé par méthode documentée |
| **M3** | `train_model.py:109-135` | Le test est utilisé **uniquement pour rapporter** (la sélection par CV est correcte [V]) mais le rapport `model_comparison.csv` ne contient que des métriques de **test**, triées par ROC-AUC de test ; ni écart-type de CV, ni AP, ni log-loss, ni temps, ni métadonnées d'exécution [V] | Impossible de distinguer validation et test ; tentation de relire le test pour choisir | CV par pli + moyenne/écart-type pour **tous** les modèles ; le test n'évalue que le modèle retenu, une seule fois |
| M4 | `train_model.py:49-61` | Seulement LR + RF ; aucun `DummyClassifier`, arbre, boosting [V] | Pas de plancher de référence ; comparaison pauvre | 5 modèles demandés, mêmes plis, même prétraitement |
| M5 | `train_model.py` | Critère de sélection : ROC-AUC en dur, aucune règle contre la sur-complexité [V] | Le modèle le plus complexe l'emporterait pour un écart insignifiant | Critère configurable (défaut : Average Precision), règle « un écart-type » préférant le modèle le plus simple |
| M6 | `data_loader.clean_data`, `pipeline.py` | **[V]** Les 11 `TotalCharges` vides correspondent **toutes** à `tenure == 0` (aucune facturation encore émise). Elles sont imputées par la médiane (~1 397 $) ; `ChargeRatio` en hérite | Valeur sémantiquement fausse (la vraie valeur est 0) ; effet faible (11 lignes sur 7 043) | Règle explicite : `TotalCharges = 0` quand `tenure == 0`, imputation médiane sinon |
| M7 | `pipeline.py:56-92` | **[V]** `TotalServices` repère les services par **sous-chaîne** (`'Online'`, `'Tech'`…) : deux colonnes quelconques « OnlineFoo », « TechnicalNotes » font passer le total de 1 à 3. `ContractRisk` remplace en silence une modalité inconnue par 2 (`fillna(2)`) ; l'ordre 3/2/1 a été choisi en regardant les taux de churn observés (légère information issue de la cible, **[H]** à effet limité). `TenureSegment` produit la chaîne `"nan"` pour `tenure < 0` ou `> 72` (**[V]**) qui devient une catégorie ordinaire | Variables dérivées fragiles et peu défendables | Liste explicite de services ; `ContractMonths` (1/12/24, neutre) ; bornes ouvertes pour `TenureSegment` ; validation amont des valeurs |
| M8 | `pipeline.py:117` | `OneHotEncoder` creux ; incompatible avec `HistGradientBoostingClassifier` [H] | Bloquerait le boosting | `sparse_output=False` (jeu de 43 colonnes) |

## 7. Risques de fuite de données

**Chemin principal : aucune fuite détectée [V].** Tout le prétraitement (imputation, mise à l'échelle,
encodage) est dans le `Pipeline`, donc ajusté dans chaque pli sur le seul train ; le split est fait
avant la CV ; le test n'intervient ni dans la sélection ni dans l'ajustement ; le nettoyage
(`strip`, conversion numérique) est ligne à ligne et ne dépend d'aucune statistique globale.

À surveiller :

| # | Risque | Statut |
|---|---|---|
| L1 | `tenure` et `TotalCharges` sont des cumuls observés **à la date de l'extraction** : pour un client parti, ils s'arrêtent à son départ. Ils portent donc une part d'information « post-résultat » (données censurées). Le jeu n'a pas de dates : on ne peut pas faire de split temporel | **[H]** non testable ici ; à documenter comme limite (DATA_CARD, MODEL_CARD) |
| L2 | `automl.py` : une colonne numérique identifiant (`customer_number`) est utilisée comme variable ; une colonne proxy parfaite de la cible (`cancel_reason_code`) est acceptée sans alerte. **[V]** ROC-AUC de test = **1,0** sur un jeu où seule cette colonne est informative | Correction : détection d'identifiants numériques, de proxys quasi parfaits, avertissements affichés |
| L3 | Importance par permutation calculée sur le **test** | Acceptable (calculée **après** la sélection) ; à nommer comme telle |
| L4 | Le rééquilibrage (SMOTE…) n'est pas utilisé | Rien à corriger ; si ajouté un jour, il devra vivre dans un `imblearn.Pipeline` |

## 8. Problèmes potentiels dans le traitement des données

Profil **[V]** du jeu `WA_Fn-UseC_-_Telco_Customer_Churn.csv` :

- 7 043 lignes × 21 colonnes ; 0 valeur manquante au sens pandas, mais `TotalCharges` est de type texte avec **11 chaînes vides** (toutes `tenure == 0`, toutes `Churn == "No"`).
- `customerID` unique (0 doublon) ; 22 lignes identiques *hors identifiant* (clients distincts, non supprimées — à raison).
- Cible : 1 869 `Yes` (26,54 %) / 5 174 `No`.
- Aucune colonne constante ; aucune incohérence entre `PhoneService` / `MultipleLines` ni entre `InternetService` / services Internet ; `MonthlyCharges` entre 18,25 et 118,75 ; `TotalCharges ≈ tenure × MonthlyCharges` (corrélation 0,9996 → forte redondance entre ces trois variables).
- Taux de churn par ancienneté : 47,4 % (0-12 mois), 28,7 %, 21,6 %, 11,9 % (36-72 mois).

Constats de gestion de la provenance :

| # | Constat | Conséquence | Correction |
|---|---|---|---|
| **G1** | **[V]** Aucun fichier `LICENSE`. Le dépôt d'origine est sous **licence MIT** (« Copyright (c) 2026 Customer Churn Prediction Project Contributors », vérifié sur `raw.githubusercontent.com/infabuilds/customer-churn-prediction/main/LICENSE`) ; `src/__init__.py` nomme « Davide Infantino » | La MIT impose de conserver la notice de copyright dans les copies substantielles | `LICENSE` (MIT) reprenant la notice d'origine **et** la vôtre, + `NOTICE.md` décrivant ce qui est repris et ce qui est nouveau |
| **G2** | **[V]** `data/*.csv` (970 Ko) est **suivi par git** ; licence du jeu **non vérifiée** (échantillon IBM diffusé via Kaggle ; aucune licence explicite trouvée chez IBM ni Kaggle ; un miroir Hugging Face l'étiquette CC-BY-4.0, ce qui ne vaut pas licence d'origine) | Redistribution publique de données dont les droits sont incertains | `data/README.md` (téléchargement manuel), CSV ignoré par git à l'avenir ; **décision à prendre avant publication** : retirer le fichier de l'index (`git rm --cached`) |
| G3 | **[V]** `venv/` (ignoré par git) a été créé pour **un autre projet** (`pyvenv.cfg` : `...\AI\llm1\04_code\venv`) | Non portable ; ne doit pas servir de référence d'installation | Instructions d'installation vérifiées à partir d'un environnement propre |
| G4 | **[V]** Un seul commit ; pas de remote `upstream` | La provenance du code ne se lit pas dans l'historique | Attribution écrite (G1) ; pas de réécriture d'historique |

## 9. Problèmes de cohérence entre entraînement et prédiction

Point fort **[V]** : un seul objet `Pipeline` est sérialisé et réutilisé (app, API) ; aucune réimplémentation
du prétraitement côté inférence. **À conserver.**

| # | Fichier | Constat | Conséquence | Correction |
|---|---|---|---|---|
| C1 | `service/main.py:68-91`, `app.py:276-310`, `pipeline.py:27-54` | Le **contrat d'entrée** est défini à trois endroits : modèle pydantic, widgets Streamlit (listes de choix, bornes 18,25–118,75), listes de colonnes du pipeline [V] | Dérive silencieuse si le jeu ou le pipeline changent [H] | Un **schéma de données** unique, déduit du jeu d'entraînement, sérialisé dans `metadata.json` et utilisé par la validation, l'API et les formulaires |
| C2 | `service/main.py:114`, `config.RISK_LEVELS`, `app.py:343` | Seuil (0,5) et bandes de risque (0,3/0,7) définis à trois endroits [V] | Résultats différents selon l'interface si l'un change | Seuil et bandes lus dans les métadonnées du modèle |
| C3 | `app.py:69`, `service/main.py:40` | `joblib.load` sans vérification d'intégrité ni de version ; `metadata.json` contient `sklearn_version` mais **personne ne la compare** ; l'API ne tolère que `FileNotFoundError` à l'import [V] | Un pickle altéré ou d'une autre version est exécuté/échoue sans message clair ; charger un pickle est de l'exécution de code | Empreinte SHA-256 dans les métadonnées vérifiée avant chargement (contrôle d'intégrité, pas d'authenticité), contrôle de version, message d'erreur explicite |
| C4 | `pipeline.py` | Pas de validation de types/valeurs avant le pipeline (voir §4) | Messages incompréhensibles pour l'utilisateur | `validate_features()` avec rapport lisible (colonnes manquantes, types, valeurs hors schéma) |

## 10. État des tests et de la CI

- **Tests : aucun** [V] (`tests/` absent ; `conftest.py` ne sert à rien tant qu'il n'y a pas de test).
- **CI : aucune** [V] (`.github/` absent). Le récit de `generate_pdf.py` mentionne pourtant une « performance gate » de CI qui n'existe pas.
- **Lint** : non configuré ; ruff 0.17.2 avec ses règles par défaut remonte 61 alertes sur le code existant [V].
- **Docker** : [V] le Dockerfile **d'origine** a été construit à partir d'une extraction du commit `1da61ba` dans un
  dossier temporaire (base `python:3.12-slim`, installation de tout le `requirements.txt` épinglé, image de 1,26 Go) ;
  le conteneur démarre, `/health` répond `ok` et `/predict` renvoie une probabilité de 0,9375 pour l'exemple de la
  documentation. Il fonctionnait donc. Observation : l'image installe Streamlit, Plotly, Matplotlib… dont l'API n'a pas besoin.

## 11. Dépendances inutiles ou redondantes

| Dépendance | Constat | Action |
|---|---|---|
| `matplotlib`, `seaborn` | Utilisées seulement par `scripts/generate_figures.py` et les notebooks, pas par l'app ni l'API [V] | Passer en dépendances de développement |
| `reportlab` | Importée par `generate_pdf.py`, **absente** du `requirements.txt` [V] | Supprimer le script |
| `pytest`, `httpx` | Dépendances de test dans le fichier d'exécution [V] | `requirements-dev.txt` |
| `streamlit`, `plotly` dans l'image de l'API | Inutiles pour l'API [V par lecture] | `requirements-api.txt` pour l'image Docker |
| `scipy`, `joblib` | Déjà tirées par scikit-learn ; `joblib` est importé directement : à garder déclarée | — |
| Versions épinglées (`pandas==3.0.3`, `numpy==2.4.6`, `scikit-learn==1.8.0`…) | **[V]** Fonctionnelles sous Python 3.14.5 (environnement vérifié) **et** sous Python 3.12 (le build Docker d'origine les installe sans erreur) | Garder les versions ; vérifier l'installation dans un environnement vierge |

## 12. Éléments à conserver

- Principe du **pipeline unique** sérialisé (`Pipeline` + `ColumnTransformer`, `handle_unknown="ignore"`).
- Séparation train/test stratifiée **avant** la CV, sélection sur le train uniquement.
- Garde-fou « données synthétiques » de `load_telco_data` (principe : jamais de données fabriquées en silence) — reformulé en vérification du jeu de démonstration.
- Contributions exactes de la régression logistique (`explain.py`) — idée conservée, implémentation fiabilisée.
- Structure de l'API (routes `/health`, `/model`, `/predict`, `/predict/batch`, limite de lot).
- Dockerfile (base `python:3.12-slim`, `HEALTHCHECK`), `.gitignore`, `.dockerignore`.
- Logique d'inférence de types / protections de `automl.py` (taille minimale, classe minoritaire, cible binaire), fusionnée dans le pipeline commun.

## 13. Éléments à modifier

| Élément | Modification |
|---|---|
| `src/` (paquet générique) | → paquet `churnguard` ; modules `data`, `schema`, `preprocessing`, `pipeline`, `training`, `evaluation`, `predictions`, `error_analysis`, `explainability`, `retention_simulator`, `config` |
| `train_model.py` | Entraînement de 5 modèles, CV par pli, sélection configurable, seuil, rapports et métadonnées reproductibles |
| `app.py` | Routeur Streamlit + 9 pages (Dashboard … Model Card) ; chemins et schéma issus des artefacts |
| `service/main.py` | Modèle d'entrée dérivé du schéma sauvegardé, seuil et bandes lus dans les métadonnées, vérification d'intégrité, erreurs claires, tests |
| `automl.py` | Fusionné dans `training.py` (une seule implémentation) ; cible et classe positive **choisies explicitement** ; détection d'identifiants et de fuites |
| `requirements.txt` | Séparé en exécution / API / développement |
| `reports/*.csv`, `models/*` | Régénérés par le nouveau protocole (les anciens chiffres ne sont plus valables : modèle recalibré) |
| `Dockerfile` | Dépendances minimales de l'API, vérification de la construction |
| Notebooks | Mis à jour sur la nouvelle API (ou retirés s'ils dupliquent le script) |

## 14. Éléments à créer

`README.md`, `LICENSE`, `NOTICE.md`, `data/README.md`, `docs/DATA_CARD.md`, `docs/METHODOLOGY.md`,
`docs/ECONOMIC_ASSUMPTIONS.md`, `docs/MODEL_CARD.md`, `configs/config.toml`, `pyproject.toml`, `tests/`
(données synthétiques identifiées comme telles), `.github/workflows/ci.yml` (ruff + pytest),
`requirements-dev.txt`, `requirements-api.txt`, modules `error_analysis`, `retention_simulator`,
`evaluation` (seuils), `schema`.

## 15. Plan d'exécution priorisé

| Étape | Contenu | Dépend de | Priorité |
|---|---|---|---|
| E1 | Socle : paquet `churnguard`, configuration, `schema`, `data` (validation et profil), licence et notice | — | P1 |
| E2 | `preprocessing` / `pipeline` (variables dérivées corrigées, 5 modèles) | E1 | P1 |
| E3 | `evaluation` (métriques, seuils, calibration) et `training` (CV, sélection, OOF, test unique, artefacts) ; `train_model.py` | E2 | P1 |
| E4 | `predictions` (bundle, intégrité, scoring, export) | E3 | P1 |
| E5 | Tests du socle (données synthétiques) | E1-E4 | P1 |
| E6 | `error_analysis`, `explainability` | E4 | P2 |
| E7 | `retention_simulator` + tests des formules | — (autonome) | P3 |
| E8 | Application Streamlit (9 sections) et entraînement sur CSV utilisateur | E4, E6, E7 | P4 |
| E9 | API FastAPI alignée, tests des routes | E4 | P2 |
| E10 | CI, ruff, Docker vérifié | E5 | P4 |
| E11 | README, DATA_CARD, METHODOLOGY, ECONOMIC_ASSUMPTIONS, MODEL_CARD (uniquement avec des chiffres **réellement calculés**) | E3-E8 | P4 |
| E12 | Vérification finale : tests, installation propre, lancement de l'app et de l'API, rapport | tout | — |

**Hors périmètre de cette version** (comme demandé) : authentification, base de données, microservices,
réentraînement planifié, SHAP (voir METHODOLOGY : dépendance non nécessaire ici, compatibilité
Python 3.14 non vérifiée).

---

## Annexe — suivi après implémentation (2026-10-26)

Rédigée à la fin du travail ; elle ne modifie pas l'audit ci-dessus, qui décrit l'état **avant** intervention.

| Constat de l'audit | Suite donnée | Preuve |
|---|---|---|
| M1 probabilités gonflées par `class_weight` | Pondération supprimée ; calibration mesurée | moyenne prédite 0,2654 / observée 0,2654 (hors-pli) |
| M2 seuil 0,5 en dur | Seuil recommandé sur prédictions hors-pli (F1) + analyse de seuils | `reports/threshold_analysis_validation.csv` |
| M3/M4/M5 comparaison pauvre, 2 modèles | 5 modèles, plis identiques, 9 métriques, écart-type, critère configurable | `reports/model_comparison.csv`, `cv_folds.csv` |
| M6/M7 variables dérivées fragiles | `TotalCharges = 0` si `tenure = 0`, liste explicite de services, `ContractMonths`, bornes ouvertes | `tests/test_preprocessing.py` |
| L2 fuite non détectée en « votre CSV » | Écran d'identifiants / proxys de la cible, sur le split d'entraînement | `tests/test_data.py`, `test_training.py` |
| C1–C4 contrat d'entrée dupliqué, seuil dupliqué, pickle non vérifié | Schéma unique sérialisé, seuil et segments dans les métadonnées, SHA-256 + version vérifiés | `tests/test_predictions.py`, `test_api.py` |
| Q1–Q8 qualité | Paquet `churnguard`, configuration unique, `logging`, exceptions explicites, Ruff | `ruff check .` : aucune alerte |
| Pas de tests / de CI | Suite de tests (données synthétiques étiquetées) ; workflow GitHub Actions | `pytest` : 228 réussis ; **CI non exécutée** (rien n'a été poussé) |
| Dockerfile | Image API seule, utilisateur non privilégié ; construite et lancée sous Python 3.12 | `/health` ok, 651 Mo (contre 1,26 Go) |
| G1 licence absente | `LICENSE` (copyright d'origine conservé) + `NOTICE.md` | — |
| G2 CSV suivi par git, licence non vérifiée | `data/README.md`, CSV ignoré pour l'avenir | **Non résolu** : le fichier reste dans l'historique git (commit `1da61ba`) — décision à prendre avant toute publication |

**Écarts et limites constatés en cours de route** (non anticipés dans l'audit) :

- Le nouveau modèle score un peu moins bien que l'ancien sur le même test (ROC-AUC 0,842 contre 0,847 ; précision
  moyenne 0,634 contre 0,665) ; la différence est dans le bruit d'un test de 1 409 lignes (intervalle de confiance à
  95 % du ROC-AUC : 0,823–0,865). Rien n'a été ajusté vers le test.
- Les quatre variables dérivées n'apportent pas de gain mesurable en validation croisée (précision moyenne 0,667 avec,
  0,661 sans, écart-type ≈ 0,02) ; elles sont conservées par choix de réutilisation, sans revendiquer d'apport.
- Un `run_id` purement numérique était relu comme un nombre par pandas (test intermittent) : préfixe `run-` ajouté.
- Pendant la session, des commits ont été ajoutés au dépôt en parallèle (dont un `README.md` rédigé ailleurs) ; ce
  README a été restauré à l'identique (voir HANDOFF).
