"""Application Streamlit : prédire si le traitement antidiabétique d'un patient sera modifié.

Modèle : XGBoost (notebook « Membre 5 — modification du traitement antidiabétique »).
Le fichier model.joblib contient {'model', 'scaler', 'meta'} exportés depuis le notebook.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# ⚠️ À CONFIRMER : la signification de ces codes n'est PAS écrite dans le notebook.
# Remplacez les textes ci-dessous quand vous connaissez le vrai sens (ex. 0: 'Femme').
LIBELLES_SEXE = {0: 'Code 0 (à confirmer)', 1: 'Code 1 (à confirmer)'}
LIBELLES_GLYCEMIE = {0: 'Code 0 (à confirmer)', 1: 'Code 1 (à confirmer)',
                     2: 'Code 2 (à confirmer)', 3: 'Code 3 (à confirmer)'}
# ---------------------------------------------------------------------------

# Codes HbA1c : signification donnée dans la version précédente du notebook de l'équipe
LIBELLES_HBA1C = {0: 'Non mesurée', 1: 'Normale', 2: 'Supérieure à 7 %', 3: 'Supérieure à 8 %'}

# Noms lisibles des catégories (les colonnes one-hot du notebook)
LIBELLES_CATEGORIES = {
    'race_': {'AfricanAmerican': 'Afro-américaine', 'Asian': 'Asiatique', 'Caucasian': 'Caucasienne',
              'Hispanic': 'Hispanique', 'Other': 'Autre'},
    'admission_type_id_': {'Urgence': 'Urgence', 'Programmee': 'Programmée', 'Autre': 'Autre'},
    'admission_source_id_': {'Urgences': 'Service des urgences', 'Medecin': 'Envoyé par un médecin',
                             'Autre': 'Autre'},
}
FAMILLES_DIAGNOSTIC = {'Circulatoire': 'Circulatoire', 'Diabete': 'Diabète', 'Respiratoire': 'Respiratoire',
                       'Digestif': 'Digestif', 'Urinaire': 'Urinaire', 'Musculaire': 'Musculaire',
                       'Blessure': 'Blessure', 'Tumeur': 'Tumeur', 'Autre': 'Autre'}
for prefixe in ['diag_1_', 'diag_2_', 'diag_3_']:
    LIBELLES_CATEGORIES[prefixe] = FAMILLES_DIAGNOSTIC

GROUPES_RISQUE = ['Faible', 'Modéré', 'Élevé', 'Très élevé']

NOMS_VARIABLES = {'A1Cresult': 'HbA1c', 'max_glu_serum': 'Glycémie maximale', 'num_medications': 'Nombre de médicaments',
                  'num_procedures': "Nombre d'actes", 'num_lab_procedures': "Nombre d'analyses",
                  'time_in_hospital': 'Durée du séjour', 'number_diagnoses': 'Nombre de diagnostics', 'age': 'Âge',
                  'gender': 'Sexe', 'number_outpatient': 'Consultations externes', 'number_emergency': 'Passages aux urgences',
                  'number_inpatient': 'Hospitalisations passées'}
PREFIXES_LISIBLES = {'diag_1_': 'Diagnostic principal', 'diag_2_': 'Diagnostic secondaire', 'diag_3_': '3ᵉ diagnostic',
                     'admission_source_id_': 'Provenance', 'admission_type_id_': "Type d'admission", 'race_': 'Origine'}


def nom_lisible(colonne):
    """Nom compréhensible d'une colonne technique (ex. diag_1_Diabete -> Diagnostic principal : Diabète)."""
    for prefixe, nom in PREFIXES_LISIBLES.items():
        if colonne.startswith(prefixe):
            categorie = colonne[len(prefixe):]
            return f'{nom} : {LIBELLES_CATEGORIES[prefixe].get(categorie, categorie)}'
    return NOMS_VARIABLES.get(colonne, colonne)


@st.cache_resource
def charger():
    """Charge le modèle, le scaler et les métadonnées (une seule fois)."""
    return joblib.load(Path(__file__).parent / 'model.joblib')


def preparer(saisie, meta):
    """Transforme la saisie du formulaire en une ligne identique aux données d'entraînement."""
    ligne = {c: 0.0 for c in meta['colonnes']}                      # toutes les colonnes, dans le bon ordre
    for nom, valeur in saisie['numeriques'].items():
        ligne[nom] = float(np.log1p(valeur)) if nom in meta['log1p'] else float(valeur)  # même log1p que data_clean
    for prefixe, categorie in saisie['categories'].items():
        ligne[prefixe + categorie] = 1.0                             # one-hot : une seule colonne à 1 par groupe
    return pd.DataFrame([ligne], columns=meta['colonnes'])


def predire(saisie, paquet):
    """Applique EXACTEMENT la même préparation que le notebook (même scaler, même ordre), puis le modèle."""
    X = preparer(saisie, paquet['meta'])
    X_std = pd.DataFrame(paquet['scaler'].transform(X), columns=X.columns)
    return float(paquet['model'].predict_proba(X_std)[0, 1])


def groupe_de_risque(proba, meta):
    """Quartiles des probabilités du jeu de test : 25 % des patients dans chaque groupe."""
    return GROUPES_RISQUE[int(np.searchsorted(meta['seuils_groupes_risque'], proba, side='right'))]


def champ_entier(libelle, colonne, meta, cle, aide=None):
    """Champ numérique entier, borné au min/max vus pendant l'entraînement."""
    s = meta['stats'][colonne]
    lo, hi, med = s['min'], s['max'], s['mediane']
    if colonne in meta['log1p']:                                     # l'utilisateur saisit la vraie valeur
        lo, hi, med = np.expm1(lo), np.expm1(hi), np.expm1(med)
    return st.number_input(libelle, min_value=int(round(lo)), max_value=int(round(hi)),
                           value=int(round(med)), step=1, key=cle, help=aide)


def liste_categorie(libelle, prefixe, meta, cle):
    """Une liste déroulante pour un groupe de colonnes one-hot."""
    groupe = meta['groupes_onehot'][prefixe]
    categories = [c[len(prefixe):] for c in groupe['colonnes']]
    defaut = groupe['plus_frequente'][len(prefixe):]
    noms = LIBELLES_CATEGORIES[prefixe]
    return st.selectbox(libelle, categories, index=categories.index(defaut),
                        format_func=lambda c: noms.get(c, c), key=cle)


def liste_codes(libelle, colonne, libelles, meta, cle):
    """Une liste déroulante pour une variable codée (0, 1, 2, 3)."""
    codes = sorted(libelles)
    defaut = int(round(meta['stats'][colonne]['mediane']))
    return st.selectbox(libelle, codes, index=codes.index(defaut), format_func=lambda c: libelles[c], key=cle)


# ============================================================================ Interface
st.set_page_config(page_title='Modification du traitement antidiabétique', page_icon='💊')
paquet = charger()
meta = paquet['meta']
m = meta['metriques_test']

st.title('💊 Modification du traitement antidiabétique')
st.write("Remplissez les informations du séjour : le modèle estime la probabilité que le **traitement "
         "antidiabétique du patient soit modifié** avant sa sortie.")

with st.sidebar:
    st.header('À propos du modèle')
    st.write('**XGBoost** entraîné sur 70 230 patients diabétiques (hôpitaux américains, 1999–2008).')
    st.write(f"Sur les patients de test : **ROC-AUC = {m['roc_auc']:.3f}**. "
             f"Au seuil {meta['seuil']:.2f}, il repère **{m['recall_seuil']:.0%}** des patients dont le "
             f"traitement a été modifié.")
    st.write('**Variables les plus importantes :**')
    for v in list(meta['importance'])[:5]:
        st.write(f'- {nom_lisible(v)}')

with st.form('formulaire'):
    st.subheader('Informations principales')
    c1, c2 = st.columns(2)
    with c1:
        hba1c = liste_codes('Résultat de l\'HbA1c', 'A1Cresult', LIBELLES_HBA1C, meta, 'hba1c')
        diag1 = liste_categorie('Diagnostic principal', 'diag_1_', meta, 'diag1')
        diag2 = liste_categorie('Diagnostic secondaire', 'diag_2_', meta, 'diag2')
        diag3 = liste_categorie('3ᵉ diagnostic', 'diag_3_', meta, 'diag3')
    with c2:
        glycemie = liste_codes('Glycémie maximale (code)', 'max_glu_serum', LIBELLES_GLYCEMIE, meta, 'glycemie')
        nb_medicaments = champ_entier('Nombre de médicaments pendant le séjour', 'num_medications', meta, 'num_med')
        nb_actes = champ_entier('Nombre d\'actes médicaux (hors analyses)', 'num_procedures', meta, 'num_proc')
        source = liste_categorie('Provenance du patient', 'admission_source_id_', meta, 'source')

    with st.expander('Autres informations'):
        c3, c4 = st.columns(2)
        with c3:
            age = st.selectbox('Tranche d\'âge', list(range(5, 100, 10)), index=6,
                               format_func=lambda a: f'{a - 5} à {a + 4} ans', key='age')
            sexe = liste_codes('Sexe (code)', 'gender', LIBELLES_SEXE, meta, 'sexe')
            origine = liste_categorie('Origine', 'race_', meta, 'origine')
            type_adm = liste_categorie('Type d\'admission', 'admission_type_id_', meta, 'type_adm')
            nb_diag = champ_entier('Nombre de diagnostics enregistrés', 'number_diagnoses', meta, 'nb_diag')
        with c4:
            duree = champ_entier('Durée du séjour (jours)', 'time_in_hospital', meta, 'duree')
            nb_analyses = champ_entier('Nombre d\'analyses de laboratoire', 'num_lab_procedures', meta, 'nb_analyses')
            consultations = champ_entier('Consultations externes (année précédente)', 'number_outpatient', meta, 'consult')
            urgences = champ_entier('Passages aux urgences (année précédente)', 'number_emergency', meta, 'urgences')
            hospit = champ_entier('Hospitalisations (année précédente)', 'number_inpatient', meta, 'hospit')

    envoyer = st.form_submit_button('Prédire', type='primary')

if envoyer:
    saisie = {
        'numeriques': {'A1Cresult': hba1c, 'max_glu_serum': glycemie, 'num_medications': nb_medicaments,
                       'num_procedures': nb_actes, 'age': age, 'gender': sexe, 'number_diagnoses': nb_diag,
                       'time_in_hospital': duree, 'num_lab_procedures': nb_analyses,
                       'number_outpatient': consultations, 'number_emergency': urgences,
                       'number_inpatient': hospit},
        'categories': {'diag_1_': diag1, 'diag_2_': diag2, 'diag_3_': diag3, 'admission_source_id_': source,
                       'race_': origine, 'admission_type_id_': type_adm},
    }
    proba = predire(saisie, paquet)
    groupe = groupe_de_risque(proba, meta)

    st.subheader('Résultat')
    r1, r2 = st.columns(2)
    r1.metric('Probabilité de modification', f'{proba:.0%}')
    r2.metric('Groupe de risque', groupe)
    if proba >= meta['seuil']:
        st.warning(f"**Modification probable** (probabilité ≥ {meta['seuil']:.2f}) : "
                   "il est conseillé de demander l'avis du diabétologue avant la sortie.")
    else:
        st.success(f"**Modification peu probable** (probabilité < {meta['seuil']:.2f}).")
    st.write(f"Le groupe de risque compare ce patient aux patients de test : chaque groupe contient 25 % "
             f"d'entre eux (« {GROUPES_RISQUE[0]} » = les 25 % de probabilités les plus basses).")
    st.info(f"**Fiabilité du modèle** (patients de test) : ROC-AUC = {m['roc_auc']:.3f}. Au seuil "
            f"{meta['seuil']:.2f}, il repère {m['recall_seuil']:.0%} des patients dont le traitement a été "
            f"modifié, mais {1 - m['precision_seuil']:.0%} de ses alertes sont fausses.")
    st.caption('⚠️ Projet étudiant : cette prédiction est une aide statistique, avec une précision modérée. '
               'Elle ne remplace pas l\'avis d\'un médecin.')
