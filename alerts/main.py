#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Système d'Alertes de Maintenance Prédictive
Auteur: Aymane ISSAMI
Organisation: Société Nationale de Radiodiffusion et de Télévision du Maroc
Description: Système d'alertes automatisé professionnel pour la prédiction de maintenance d'équipements
Version: 1.0

"""

import smtplib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import joblib
import pickle
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
import warnings
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
warnings.filterwarnings('ignore')

# Palette de couleurs officielle SNRT
SNRT_COLORS = {
    'primary_blue': '#1e4a72',      # Bleu principal SNRT
    'accent_red': '#c41e3a',        # Rouge SNRT
    'accent_orange': '#e67e22',     # Orange SNRT
    'success_green': '#27ae60',     # Vert de succès
    'warning_yellow': '#f39c12',    # Jaune d'avertissement
    'light_gray': '#f8fafc',        # Gris clair d'arrière-plan
    'dark_gray': '#2c3e50',         # Gris foncé pour le texte
    'white': '#ffffff',             # Arrière-plan blanc
    'critical_bg': '#fdf2f2',       # Arrière-plan critique
    'high_bg': '#fef5e7',          # Arrière-plan priorité élevée
    'moderate_bg': '#fff8f0',       # Arrière-plan priorité modérée
    'low_bg': '#f0fff4'            # Arrière-plan priorité faible
}

# Configuration SMTP
SMTP_CONFIG = {
    'server': 'smtp.gmail.com',
    'port': 587,
    'user': 'webxcelsite@gmail.com',
    'password': 'lrrf lcih rofh lbix',
    'sender_name': 'Système de Maintenance Prédictive SNRT'
}

# Configuration des destinataires d'emails
RECIPIENTS = {
    'critique': ['issami.aymane@gmail.com', 'maintenance.critical@snrt.ma'],
    'eleve': ['issami.aymane@gmail.com', 'maintenance.high@snrt.ma'],
    'modere': ['issami.aymane@gmail.com', 'maintenance.moderate@snrt.ma'],
    'rapports': ['issami.aymane@gmail.com']
}

# Configuration des logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alerts.log'),
        logging.StreamHandler()
    ]
)

def obtenir_chemin_modele():
    """Configuration interactive du chemin du modèle"""
    print("\n" + "="*60)
    print("CONFIGURATION DU MODÈLE PRÉDICTIF SNRT")
    print("="*60)
    print("Veuillez spécifier le chemin vers votre fichier de modèle (.pkl)")
    
    while True:
        chemin_fichier = input("\nEntrez le chemin vers votre fichier de modèle (.pkl): ").strip()
        
        if os.path.exists(chemin_fichier):
            if chemin_fichier.endswith('.pkl'):
                return chemin_fichier
            else:
                print("Le fichier doit avoir l'extension .pkl")
                continue
        else:
            print(f"Fichier non trouvé: {chemin_fichier}")
            continue

def rechercher_fichiers_modele():
    """Rechercher les fichiers de modèle .pkl dans le répertoire courant"""
    fichiers_modele = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pkl') and 'model' in file.lower():
                fichiers_modele.append(os.path.join(root, file))
    return fichiers_modele

def obtenir_source_donnees():
    """Configuration interactive de la source de données"""
    print("\n" + "="*60)
    print("CONFIGURATION DE LA SOURCE DE DONNÉES SNRT")
    print("="*60)
    print("Comment souhaitez-vous charger les données d'équipement?")
    print("\nOptions disponibles:")
    print("  1. Fichier CSV (.csv)")
    print("  2. Fichier Excel (.xlsx)")
    print("  3. Connexion base de données")
    print("  4. API REST")
    
    while True:
        choix = input("\nVotre choix (1-4): ").strip()
        
        if choix == "1":
            return obtenir_chemin_csv()
        elif choix == "2":
            return obtenir_chemin_excel()
        elif choix == "3":
            return configurer_connexion_bdd()
        elif choix == "4":
            return configurer_connexion_api()
        else:
            print("Choix invalide. Veuillez entrer 1, 2, 3 ou 4.")

def obtenir_chemin_csv():
    """Configuration du chemin du fichier CSV"""
    while True:
        chemin_fichier = input("\nEntrez le chemin vers votre fichier CSV: ").strip()
        if os.path.exists(chemin_fichier) and chemin_fichier.endswith('.csv'):
            return {'type': 'csv', 'chemin': chemin_fichier}
        else:
            print("Fichier CSV non trouvé ou extension incorrecte")

def obtenir_chemin_excel():
    """Configuration du chemin du fichier Excel"""
    while True:
        chemin_fichier = input("\nEntrez le chemin vers votre fichier Excel: ").strip()
        if os.path.exists(chemin_fichier) and chemin_fichier.endswith(('.xlsx', '.xls')):
            return {'type': 'excel', 'chemin': chemin_fichier}
        else:
            print("Fichier Excel non trouvé ou extension incorrecte")

def configurer_connexion_bdd():
    """Configuration de la connexion base de données"""
    print("\nConfiguration de la base de données:")
    config_bdd = {
        'type': 'database',
        'host': input("Hôte de la base de données: ").strip(),
        'port': input("Port (défaut 5432): ").strip() or "5432",
        'database': input("Nom de la base de données: ").strip(),
        'username': input("Nom d'utilisateur: ").strip(),
        'password': input("Mot de passe: ").strip(),
        'table': input("Nom de la table des équipements: ").strip()
    }
    return config_bdd

def configurer_connexion_api():
    """Configuration de la connexion API"""
    print("\nConfiguration de l'API:")
    config_api = {
        'type': 'api',
        'base_url': input("URL de base de l'API: ").strip(),
        'endpoint': input("Endpoint des équipements: ").strip(),
        'api_key': input("Clé API (optionnel): ").strip(),
        'headers': {}
    }
    
    if config_api['api_key']:
        config_api['headers']['Authorization'] = f"Bearer {config_api['api_key']}"
    
    return config_api

class ServiceEmailMaintenance:
    """Service d'email professionnel pour les alertes de maintenance"""
    
    def __init__(self, config_smtp: Dict):
        self.config_smtp = config_smtp
        self.logger = logging.getLogger(__name__)
        
    def envoyer_email(self, emails_destinataires: List[str], sujet: str, 
                     contenu_html: str, pieces_jointes: List[str] = None) -> bool:
        """
        Envoyer un email HTML professionnel avec pièces jointes optionnelles
        
        Args:
            emails_destinataires: Liste des adresses email des destinataires
            sujet: Ligne d'objet de l'email
            contenu_html: Contenu HTML de l'email
            pieces_jointes: Liste des chemins de fichiers à joindre
        
        Returns:
            bool: True si envoyé avec succès, False sinon
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['From'] = f"{self.config_smtp['sender_name']} <{self.config_smtp['user']}>"
            msg['To'] = ', '.join(emails_destinataires)
            msg['Subject'] = sujet
            
            partie_html = MIMEText(contenu_html, 'html', 'utf-8')
            msg.attach(partie_html)
            
            if pieces_jointes:
                for chemin_fichier in pieces_jointes:
                    if os.path.exists(chemin_fichier):
                        self._joindre_fichier(msg, chemin_fichier)
            
            with smtplib.SMTP(self.config_smtp['server'], self.config_smtp['port']) as server:
                server.starttls()
                server.login(self.config_smtp['user'], self.config_smtp['password'])
                texte = msg.as_string()
                server.sendmail(self.config_smtp['user'], emails_destinataires, texte)
            
            self.logger.info(f"Email envoyé avec succès à: {', '.join(emails_destinataires)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Échec de l'envoi d'email: {str(e)}")
            return False
    
    def _joindre_fichier(self, msg: MIMEMultipart, chemin_fichier: str):
        """Joindre un fichier au message email"""
        try:
            with open(chemin_fichier, "rb") as piece_jointe:
                partie = MIMEBase('application', 'octet-stream')
                partie.set_payload(piece_jointe.read())
            
            encoders.encode_base64(partie)
            partie.add_header(
                'Content-Disposition',
                f'attachment; filename= {os.path.basename(chemin_fichier)}'
            )
            msg.attach(partie)
        except Exception as e:
            self.logger.warning(f"Impossible de joindre le fichier {chemin_fichier}: {str(e)}")

class PredicteurMaintenance:
    """Prédicteur de maintenance professionnel avec logique de filtrage Streamlit"""
    
    def __init__(self, chemin_modele: str, df_propre: pd.DataFrame = None):
        self.package_modele = None
        self.df_propre = df_propre
        self.model_df = None  # Dataset enrichi avec features temporelles
        self.valid_equipments = None  # Équipements avec au moins 2 pannes
        self.logger = logging.getLogger(__name__)
        
        if not self.charger_modele(chemin_modele):
            raise Exception(f"Échec du chargement du modèle: {chemin_modele}")
        
        # Créer les features temporelles après validation des données
        if self.df_propre is not None:
            self._create_temporal_features()
    
    def charger_modele(self, fichier_modele: str) -> bool:
        """Charger le modèle de prédiction depuis le fichier pickle"""
        try:
            try:
                with open(fichier_modele, 'rb') as f:
                    self.package_modele = pickle.load(f)
            except Exception:
                try:
                    self.package_modele = joblib.load(fichier_modele)
                except Exception as e:
                    raise Exception(f"Impossible de charger le modèle: {e}")
            
            if not isinstance(self.package_modele, dict):
                if hasattr(self.package_modele, 'predict'):
                    modele_temp = self.package_modele
                    self.package_modele = {
                        'regression_model': modele_temp,
                        'model_name': type(modele_temp).__name__,
                        'feature_columns': [],
                        'performance_metrics': {}
                    }
                else:
                    raise Exception("Format de fichier de modèle invalide")
            
            self.logger.info(f"Modèle chargé avec succès: {self.package_modele.get('model_name', 'Inconnu')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur de chargement du modèle: {str(e)}")
            return False
    
    def _create_temporal_features(self):
        """
        Créer le dataset avec features temporelles (logique identique à Streamlit)
        """
        enhanced_data = []
        self.valid_equipments = []
        
        self.logger.info("Création des features temporelles...")
        
        for equipment in self.df_propre['code_equipement'].unique():
            eq_data = self.df_propre[self.df_propre['code_equipement'] == equipment].copy()
            
            # FILTRER UNIQUEMENT LES PANNES
            pannes = eq_data[eq_data['is_panne'] == 1].copy()
            
            # CONDITION STRICTE: Au moins 2 pannes
            if len(pannes) >= 2:
                self.valid_equipments.append(equipment)
                
                # Pour chaque panne (sauf la dernière)
                for idx in range(len(pannes) - 1):
                    current_date = pannes.iloc[idx]['DateDebut']
                    next_failure_date = pannes.iloc[idx + 1]['DateDebut']
                    
                    # Calculer days_since_last_intervention
                    past_interventions = eq_data[eq_data['DateDebut'] < current_date]
                    if len(past_interventions) > 0:
                        days_since_last = (current_date - past_interventions['DateDebut'].max()).days
                    else:
                        days_since_last = 0
                    
                    # Créer les features
                    features = {
                        'code_equipement': equipment,
                        'equipement_type': pannes.iloc[idx]['equipement'],
                        'date_actuelle': current_date,
                        'days_to_next_failure': (next_failure_date - current_date).days,
                        
                        # Historique
                        'age_equipment_days': (current_date - eq_data['DateDebut'].min()).days,
                        'total_pannes_before': len(pannes[pannes['DateDebut'] < current_date]),
                        'total_interventions_before': len(eq_data[eq_data['DateDebut'] < current_date]),
                        
                        # Activité récente
                        'interventions_last_7d': len(eq_data[
                            (eq_data['DateDebut'] < current_date) &
                            (eq_data['DateDebut'] >= current_date - pd.Timedelta(days=7))
                        ]),
                        'interventions_last_30d': len(eq_data[
                            (eq_data['DateDebut'] < current_date) &
                            (eq_data['DateDebut'] >= current_date - pd.Timedelta(days=30))
                        ]),
                        'interventions_last_90d': len(eq_data[
                            (eq_data['DateDebut'] < current_date) &
                            (eq_data['DateDebut'] >= current_date - pd.Timedelta(days=90))  # CORRECTION: DateDebut au lieu de DateDebusi
                        ]),
                        'days_since_last_intervention': days_since_last,
                        
                        # Saisonnalité 
                        'month': current_date.month,
                        'quarter': current_date.quarter,
                        'day_of_week': current_date.dayofweek,
                        'is_weekend': int(current_date.dayofweek >= 5)
                    }
                    
                    # Calculer le taux d'intervention
                    if features['age_equipment_days'] > 0:
                        features['intervention_rate'] = features['total_interventions_before'] / features['age_equipment_days']
                    else:
                        features['intervention_rate'] = 0
                    
                    enhanced_data.append(features)
        
        # Créer le DataFrame enrichi
        self.model_df = pd.DataFrame(enhanced_data)
        
        # Log pour vérification détaillée
        self.logger.info(f"Dataset enrichi créé: {len(self.model_df):,} observations temporelles")
        self.logger.info(f"Équipements valides (≥2 pannes): {len(self.valid_equipments)} sur {self.df_propre['code_equipement'].nunique()}")
        
        # NOUVEAU: Log détaillé des équipements exclus
        if len(self.valid_equipments) < self.df_propre['code_equipement'].nunique():
            equipements_exclus = self.df_propre['code_equipement'].nunique() - len(self.valid_equipments)
            taux_exclusion = (equipements_exclus / self.df_propre['code_equipement'].nunique()) * 100
            self.logger.warning(f"Équipements exclus: {equipements_exclus} ({taux_exclusion:.1f}%) - critère: <2 pannes")
            
            # Log des équipements exclus pour debugging
            tous_equipements = set(self.df_propre['code_equipement'].unique())
            equipements_exclus_list = tous_equipements - set(self.valid_equipments)
            if len(equipements_exclus_list) <= 10:  # Afficher seulement si pas trop nombreux
                self.logger.debug(f"Équipements exclus: {', '.join(list(equipements_exclus_list)[:10])}")
    
    def predire_panne(self, code_equipement: str, date_courante: datetime = None) -> Dict:
        """Prédire la panne pour un équipement spécifique"""
        if self.package_modele is None or self.df_propre is None:
            return {'erreur': 'Modèle ou données non chargés'}
        
        # NOUVEAU: Double vérification de la validité
        if self.valid_equipments is None:
            self.logger.warning("Liste des équipements valides non initialisée")
            return {'erreur': 'Système non initialisé correctement'}
        
        # Vérifier si l'équipement est dans la liste des équipements valides
        if code_equipement not in self.valid_equipments:
            return {'erreur': f'Équipement {code_equipement} non disponible pour prédiction (moins de 2 pannes historiques)'}
            
        if date_courante is None:
            date_courante = pd.Timestamp.now()
        else:
            date_courante = pd.to_datetime(date_courante)
        
        historique_eq = self.df_propre[self.df_propre['code_equipement'] == code_equipement].copy()
        if len(historique_eq) == 0:
            return {'erreur': 'Équipement non trouvé'}
        
        try:
            interventions_passees = historique_eq[historique_eq['DateDebut'] < date_courante]
            if len(interventions_passees) > 0:
                jours_depuis_derniere = (date_courante - interventions_passees['DateDebut'].max()).days
            else:
                jours_depuis_derniere = 0
            
            caracteristiques = {
                'age_equipment_days': max(0, (date_courante - historique_eq['DateDebut'].min()).days),
                'total_pannes_before': len(historique_eq[(historique_eq.get('is_panne', 0) == 1) & (historique_eq['DateDebut'] < date_courante)]),
                'total_interventions_before': len(historique_eq[historique_eq['DateDebut'] < date_courante]),
                'interventions_last_7d': len(historique_eq[(historique_eq['DateDebut'] < date_courante) &
                                                        (historique_eq['DateDebut'] >= date_courante - pd.Timedelta(days=7))]),
                'interventions_last_30d': len(historique_eq[(historique_eq['DateDebut'] < date_courante) &
                                                         (historique_eq['DateDebut'] >= date_courante - pd.Timedelta(days=30))]),
                'interventions_last_90d': len(historique_eq[(historique_eq['DateDebut'] < date_courante) &
                                                         (historique_eq['DateDebut'] >= date_courante - pd.Timedelta(days=90))]),
                'days_since_last_intervention': jours_depuis_derniere,
                'month': date_courante.month,
                'quarter': date_courante.quarter,
                'day_of_week': date_courante.dayofweek,
                'is_weekend': int(date_courante.dayofweek >= 5),
                'equipement_type_encoded': 0
            }
            
            if 'label_encoder' in self.package_modele and 'equipement' in historique_eq.columns:
                try:
                    type_eq = historique_eq['equipement'].iloc[0]
                    if hasattr(self.package_modele['label_encoder'], 'classes_'):
                        if type_eq in self.package_modele['label_encoder'].classes_:
                            caracteristiques['equipement_type_encoded'] = self.package_modele['label_encoder'].transform([type_eq])[0]
                        else:
                            caracteristiques['equipement_type_encoded'] = 0
                except Exception:
                    caracteristiques['equipement_type_encoded'] = 0
            
            if caracteristiques['age_equipment_days'] > 0:
                caracteristiques['intervention_rate'] = caracteristiques['total_interventions_before'] / caracteristiques['age_equipment_days']
            else:
                caracteristiques['intervention_rate'] = 0.0
            
            colonnes_caracteristiques = self.package_modele.get('feature_columns', list(caracteristiques.keys()))
            
            for col in colonnes_caracteristiques:
                if col not in caracteristiques:
                    caracteristiques[col] = 0.0
            
            X_pred = pd.DataFrame([caracteristiques])[colonnes_caracteristiques]
            X_pred = X_pred.fillna(0)
            X_pred = X_pred.replace([np.inf, -np.inf], 0)
            
            modele = self.package_modele['regression_model']
            jours_avant_panne = modele.predict(X_pred)[0]
            jours_avant_panne = max(0, jours_avant_panne)
            
            date_prevue = date_courante + pd.Timedelta(days=int(jours_avant_panne))
            
            # Intervalle de confiance
            intervalle_confiance = {
                'lower': max(0, int(jours_avant_panne * 0.7)),
                'upper': int(jours_avant_panne * 1.3)
            }
            
            if jours_avant_panne <= 7:
                niveau_risque = "CRITIQUE"
                classe_risque = "critique"
                priorite = "critique"
                action = "Maintenance immédiate requise"
            elif jours_avant_panne <= 30:
                niveau_risque = "ÉLEVÉ"
                classe_risque = "eleve"
                priorite = "eleve"
                action = "Planifier maintenance dans les 2 semaines"
            elif jours_avant_panne <= 90:
                niveau_risque = "MODÉRÉ"
                classe_risque = "modere"
                priorite = "modere"
                action = "Surveillance accrue recommandée"
            else:
                niveau_risque = "FAIBLE"
                classe_risque = "faible"
                priorite = "faible"
                action = "Maintenance préventive standard"
            
            return {
                'equipment_code': code_equipement,
                'equipment_name': historique_eq['equipement'].iloc[0] if 'equipement' in historique_eq.columns else 'Inconnu',
                'equipment_type': historique_eq['equipement'].iloc[0] if 'equipement' in historique_eq.columns else 'Inconnu',
                'location': historique_eq.get('localisation', pd.Series(['N/A'])).iloc[0] if 'localisation' in historique_eq.columns else 'N/A',
                'current_date': date_courante.strftime('%Y-%m-%d'),
                'predicted_failure_date': date_prevue.strftime('%Y-%m-%d'),
                'days_to_failure': int(jours_avant_panne),
                'risk_level': niveau_risque,
                'risk_class': classe_risque,
                'priority': priorite,
                'recommended_action': action,
                'confidence_interval': intervalle_confiance,
                'features': caracteristiques,
                'total_interventions': len(historique_eq),
                'last_intervention': historique_eq['DateDebut'].max().strftime('%Y-%m-%d') if len(historique_eq) > 0 else 'N/A'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur de prédiction pour {code_equipement}: {str(e)}")
            return {'erreur': f'Échec de la prédiction: {str(e)}'}
    
    def prediction_lot_securisee(self, codes_equipement: List[str], erreurs_max: int = 10) -> List[Dict]:
        """
        Prédiction par lot avec gestion d'erreurs
        MAINTENANT: Filtre automatiquement les équipements non valides
        """
        predictions = []
        compteur_erreurs = 0
        
        # Filtrer les équipements valides
        if self.valid_equipments is not None:
            codes_valides = [eq for eq in codes_equipement if eq in self.valid_equipments]
            codes_exclus = len(codes_equipement) - len(codes_valides)
            if codes_exclus > 0:
                self.logger.warning(f"{codes_exclus} équipements exclus (moins de 2 pannes historiques)")
        else:
            codes_valides = codes_equipement
        
        total_codes = len(codes_valides)
        
        self.logger.info(f"Début des prédictions par lot pour {total_codes} équipements valides")
        
        for i, code_eq in enumerate(codes_valides):
            try:
                pred = self.predire_panne(code_eq)
                if pred and 'erreur' not in pred:
                    predictions.append(pred)
                else:
                    compteur_erreurs += 1
                    self.logger.warning(f"Erreur de prédiction pour {code_eq}: {pred.get('erreur', 'Inconnue')}")
                    if compteur_erreurs >= erreurs_max:
                        self.logger.warning(f"Trop d'erreurs de prédiction ({compteur_erreurs}). Arrêt du traitement.")
                        break
            except Exception as e:
                compteur_erreurs += 1
                self.logger.error(f"Exception lors de la prédiction pour {code_eq}: {str(e)}")
                if compteur_erreurs >= erreurs_max:
                    self.logger.warning(f"Trop d'erreurs de prédiction ({compteur_erreurs}). Arrêt du traitement.")
                    break
            
            if (i + 1) % 50 == 0 or i == total_codes - 1:
                progres = (i + 1) / total_codes * 100
                self.logger.info(f"Progrès: {i + 1}/{total_codes} équipements valides ({progres:.1f}%)")
        
        self.logger.info(f"Prédictions par lot terminées: {len(predictions)} réussies, {compteur_erreurs} erreurs")
        return predictions
    
    def predire_panne_equipement(self, donnees_equipement: pd.DataFrame, 
                                 nb_equipements: int = None) -> List[Dict]:
        """
        Méthode de prédiction principale compatible avec l'interface héritée
        MAINTENANT: Utilise seulement les équipements valides
        """
        try:
            if self.df_propre is None:
                self.df_propre = donnees_equipement.copy()
                
                if 'is_panne' not in self.df_propre.columns and 'TypeIntervention' in self.df_propre.columns:
                    self.df_propre['is_panne'] = (self.df_propre['TypeIntervention'] == 'incident').astype(int)
                
                if 'DateDebut' in self.df_propre.columns:
                    self.df_propre['DateDebut'] = pd.to_datetime(self.df_propre['DateDebut'], errors='coerce')
                
                # Créer les features temporelles
                self._create_temporal_features()
            
            # NOUVEAU: Validation supplémentaire
            if self.valid_equipments is None or len(self.valid_equipments) == 0:
                self.logger.error("Aucun équipement valide trouvé après création des features temporelles")
                self.logger.error("Vérifiez:")
                self.logger.error("  - Que la colonne 'is_panne' est correctement définie")
                self.logger.error("  - Que les données contiennent des incidents (TypeIntervention='incident')")
                self.logger.error("  - Qu'il y a suffisamment d'historique de pannes par équipement")
                return []
            
            equipements_uniques = self.valid_equipments.copy()
            
            if nb_equipements and nb_equipements < len(equipements_uniques):
                equipements_uniques = equipements_uniques[:nb_equipements]
            
            self.logger.info(f"Analyse de {len(equipements_uniques)} équipements valides")
            
            predictions = self.prediction_lot_securisee(equipements_uniques)
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Erreur dans predire_panne_equipement: {str(e)}")
            return []
    
    def analyser_donnees_pour_debug(self) -> Dict:
        """Méthode d'analyse pour debugging - retourne des statistiques détaillées"""
        if self.df_propre is None:
            return {'erreur': 'Données non chargées'}
        
        stats = {
            'total_equipements': self.df_propre['code_equipement'].nunique(),
            'total_interventions': len(self.df_propre),
            'total_pannes': self.df_propre['is_panne'].sum() if 'is_panne' in self.df_propre.columns else 0,
            'taux_pannes': self.df_propre['is_panne'].mean() * 100 if 'is_panne' in self.df_propre.columns else 0,
            'periode_donnees': {
                'debut': self.df_propre['DateDebut'].min().strftime('%Y-%m-%d') if 'DateDebut' in self.df_propre.columns else 'N/A',
                'fin': self.df_propre['DateDebut'].max().strftime('%Y-%m-%d') if 'DateDebut' in self.df_propre.columns else 'N/A'
            },
            'equipements_valides': len(self.valid_equipments) if self.valid_equipments else 0,
            'observations_temporelles': len(self.model_df) if self.model_df is not None else 0
        }
        
        # Analyse par équipement
        if 'is_panne' in self.df_propre.columns:
            pannes_par_equipement = self.df_propre.groupby('code_equipement')['is_panne'].sum().reset_index()
            stats['distribution_pannes'] = {
                '0_pannes': len(pannes_par_equipement[pannes_par_equipement['is_panne'] == 0]),
                '1_panne': len(pannes_par_equipement[pannes_par_equipement['is_panne'] == 1]),
                '2_ou_plus': len(pannes_par_equipement[pannes_par_equipement['is_panne'] >= 2]),
            }
        
        return stats

class GenerateurTemplateEmail:
    """Générateur de templates email HTML professionnels avec style SNRT"""
    
    @staticmethod
    def generer_email_alerte(predictions: List[Dict], type_alerte: str) -> str:
        """Générer le contenu email HTML professionnel pour les alertes de maintenance"""
        
        predictions_filtrees = [p for p in predictions if p['priority'] == type_alerte]
        
        if not predictions_filtrees:
            return None
        
        configs_alerte = {
            'critique': {
                'titre': 'ALERTE CRITIQUE - Maintenance Immédiate Requise',
                'couleur': SNRT_COLORS['accent_red'],
                'couleur_bg': SNRT_COLORS['critical_bg']
            },
            'eleve': {
                'titre': 'ALERTE PRIORITÉ ÉLEVÉE - Maintenance Urgente',
                'couleur': SNRT_COLORS['accent_orange'],
                'couleur_bg': SNRT_COLORS['high_bg']
            },
            'modere': {
                'titre': 'ALERTE MODÉRÉE - Planification de Maintenance Requise',
                'couleur': SNRT_COLORS['warning_yellow'],
                'couleur_bg': SNRT_COLORS['moderate_bg']
            },
            'faible': {
                'titre': 'INFORMATION - Surveillance Continue',
                'couleur': SNRT_COLORS['success_green'],
                'couleur_bg': SNRT_COLORS['low_bg']
            }
        }
        
        config = configs_alerte.get(type_alerte, configs_alerte['modere'])
        
        # Calculer les statistiques pour le dashboard
        total_equipements = len(predictions_filtrees)
        jours_tous = [p['days_to_failure'] for p in predictions_filtrees]
        jours_min = min(jours_tous)
        jours_max = max(jours_tous)
        jours_moyen = np.mean(jours_tous)
        
        # Analyse par type d'équipement
        types_equipement = {}
        for pred in predictions_filtrees:
            eq_type = pred.get('equipment_type', 'Inconnu')
            if eq_type not in types_equipement:
                types_equipement[eq_type] = []
            types_equipement[eq_type].append(pred['days_to_failure'])
        
        contenu_html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{config['titre']}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {SNRT_COLORS['light_gray']};
                    line-height: 1.6;
                    color: {SNRT_COLORS['dark_gray']};
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: {SNRT_COLORS['white']};
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .main-header {{
                    background: linear-gradient(135deg, {SNRT_COLORS['primary_blue']} 0%, {SNRT_COLORS['accent_red']} 50%, {SNRT_COLORS['accent_orange']} 100%);
                    padding: 2rem;
                    border-radius: 15px 15px 0 0;
                    color: white;
                    text-align: center;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                }}
                .main-header h1 {{
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .main-header p {{
                    font-size: 1.2rem;
                    opacity: 0.9;
                    margin: 0;
                }}
                .content {{
                    padding: 40px;
                }}
                .metric-card {{
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    border-left: 5px solid {SNRT_COLORS['primary_blue']};
                    margin-bottom: 1rem;
                    transition: transform 0.2s ease;
                }}
                .metric-card:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 20px rgba(0,0,0,0.15);
                }}
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 25px;
                    margin: 40px 0;
                }}
                .metric-card-dashboard {{
                    background: linear-gradient(135deg, {SNRT_COLORS['white']} 0%, {config['couleur_bg']} 100%);
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                    text-align: center;
                    border-left: 5px solid {config['couleur']};
                    transition: transform 0.2s ease;
                }}
                .metric-number {{
                    font-size: 42px;
                    font-weight: bold;
                    margin: 15px 0;
                    color: {config['couleur']};
                }}
                .metric-label {{
                    color: {SNRT_COLORS['dark_gray']};
                    font-size: 14px;
                    text-transform: uppercase;
                    font-weight: 600;
                    margin-bottom: 5px;
                }}
                .metric-subtitle {{
                    color: {SNRT_COLORS['dark_gray']};
                    font-size: 12px;
                    opacity: 0.7;
                }}
                .page-header {{
                    background: linear-gradient(135deg, {SNRT_COLORS['light_gray']} 0%, #e2e8f0 100%);
                    padding: 1.5rem;
                    border-radius: 12px;
                    margin-bottom: 2rem;
                    border-left: 5px solid {SNRT_COLORS['primary_blue']};
                }}
                .page-header h2 {{
                    color: {SNRT_COLORS['primary_blue']};
                    margin-bottom: 0.5rem;
                    font-size: 2rem;
                    font-weight: 600;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 30px 0;
                    background-color: {SNRT_COLORS['white']};
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
                }}
                th {{
                    background-color: {SNRT_COLORS['primary_blue']};
                    color: {SNRT_COLORS['white']};
                    padding: 18px 15px;
                    text-align: left;
                    font-weight: 600;
                    font-size: 14px;
                    text-transform: uppercase;
                }}
                td {{
                    padding: 15px;
                    border-bottom: 1px solid {SNRT_COLORS['light_gray']};
                    font-size: 14px;
                }}
                .priority-critique {{
                    background-color: {SNRT_COLORS['critical_bg']};
                    font-weight: 600;
                }}
                .priority-eleve {{
                    background-color: {SNRT_COLORS['high_bg']};
                }}
                .risk-critical {{
                    background: linear-gradient(135deg, {SNRT_COLORS['accent_red']} 0%, #e74c3c 100%);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
                    display: inline-block;
                }}
                .risk-high {{
                    background: linear-gradient(135deg, {SNRT_COLORS['accent_orange']} 0%, #f39c12 100%);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    box-shadow: 0 4px 12px rgba(230, 126, 34, 0.3);
                    display: inline-block;
                }}
                .risk-moderate {{
                    background: linear-gradient(135deg, {SNRT_COLORS['warning_yellow']} 0%, #f1c40f 100%);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    box-shadow: 0 4px 12px rgba(241, 196, 15, 0.3);
                    display: inline-block;
                }}
                .risk-low {{
                    background: linear-gradient(135deg, {SNRT_COLORS['success_green']} 0%, #2ecc71 100%);
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    text-align: center;
                    font-weight: bold;
                    box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
                    display: inline-block;
                }}
                .section {{
                    margin: 50px 0;
                    padding: 40px;
                    background: {SNRT_COLORS['light_gray']};
                    border-radius: 12px;
                    border-left: 5px solid {SNRT_COLORS['primary_blue']};
                }}
                .section h2 {{
                    color: {SNRT_COLORS['primary_blue']};
                    margin-top: 0;
                    font-size: 24px;
                    font-weight: 600;
                }}
                .actions-section {{
                    margin: 40px 0;
                    padding: 30px;
                    background: {config['couleur_bg']};
                    border-radius: 12px;
                    border-left: 5px solid {config['couleur']};
                }}
                .actions-section h3 {{
                    color: {config['couleur']};
                    margin-top: 0;
                    font-size: 20px;
                }}
                .actions-section ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                .actions-section li {{
                    margin: 10px 0;
                    font-weight: 500;
                }}
                .footer {{
                    text-align: center;
                    padding: 40px;
                    background-color: {SNRT_COLORS['light_gray']};
                    border-top: 3px solid {SNRT_COLORS['primary_blue']};
                    color: {SNRT_COLORS['dark_gray']};
                }}
                .footer p {{
                    margin: 8px 0;
                }}
                .company-name {{
                    font-size: 18px;
                    font-weight: 700;
                    color: {SNRT_COLORS['primary_blue']};
                }}
                .color-dot {{
                    width: 20px;
                    height: 20px;
                    border-radius: 50%;
                    display: inline-block;
                    margin: 0 5px;
                }}
                .color-blue {{ background-color: {SNRT_COLORS['primary_blue']}; }}
                .color-red {{ background-color: {SNRT_COLORS['accent_red']}; }}
                .color-orange {{ background-color: {SNRT_COLORS['accent_orange']}; }}
                .color-green {{ background-color: {SNRT_COLORS['success_green']}; }}
                .snrt-colors {{
                    display: flex;
                    justify-content: center;
                    gap: 10px;
                    margin-bottom: 1rem;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="main-header">
                    <div class="snrt-colors">
                        <div class="color-dot color-blue"></div>
                        <div class="color-dot color-red"></div>
                        <div class="color-dot color-orange"></div>
                        <div class="color-dot color-green"></div>
                    </div>
                    <h1>Système d'Alertes de Maintenance Prédictive</h1>
                    <p>Société Nationale de Radiodiffusion et de Télévision</p>
                </div>
                
                <div class="content">
                    <div class="page-header">
                        <h2>{config['titre']}</h2>
                        <p>Analyse générée le {datetime.now().strftime('%d %B %Y à %H:%M')}</p>
                    </div>
                    
                    <div class="dashboard-grid">
                        <div class="metric-card-dashboard">
                            <div class="metric-label">Total Équipements</div>
                            <div class="metric-number">{total_equipements}</div>
                            <div class="metric-subtitle">Nécessitent attention</div>
                        </div>
                        <div class="metric-card-dashboard">
                            <div class="metric-label">Jours Minimum</div>
                            <div class="metric-number">{jours_min}</div>
                            <div class="metric-subtitle">Le plus urgent</div>
                        </div>
                        <div class="metric-card-dashboard">
                            <div class="metric-label">Jours Maximum</div>
                            <div class="metric-number">{jours_max}</div>
                            <div class="metric-subtitle">Le moins urgent</div>
                        </div>
                        <div class="metric-card-dashboard">
                            <div class="metric-label">Jours Moyen</div>
                            <div class="metric-number">{jours_moyen:.1f}</div>
                            <div class="metric-subtitle">Délai moyen</div>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Vue d'Ensemble des Prédictions</h2>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 30px;">
                            <div>
                                <h4>Analyse Temporelle</h4>
                                <p><strong>Période d'analyse:</strong> {datetime.now().strftime('%B %Y')}</p>
                                <p><strong>Équipements analysés:</strong> {total_equipements}</p>
                                <p><strong>Niveau de priorité:</strong> {type_alerte.upper()}</p>
                            </div>
                            <div>
                                <h4>Indicateurs Clés</h4>
                                <p><strong>Délai moyen:</strong> {jours_moyen:.1f} jours</p>
                                <p><strong>Équipement le plus urgent:</strong> {jours_min} jour(s)</p>
                                <p><strong>Plage de délais:</strong> {jours_min}-{jours_max} jours</p>
                            </div>
        """
        
        # Ajouter l'analyse par type d'équipement si disponible
        if types_equipement:
            contenu_html += """
                            <div>
                                <h4>Types d'Équipements</h4>
            """
            for eq_type, jours_list in list(types_equipement.items())[:3]:
                moyenne_type = np.mean(jours_list)
                contenu_html += f"""
                                <p><strong>{eq_type}:</strong> {len(jours_list)} équipement(s) - {moyenne_type:.1f}j</p>
                """
            contenu_html += "</div>"
        
        # NOUVEAU: Section des critères de sélection
        contenu_html += f"""
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Critères de Sélection des Équipements</h2>
                        <div style="background: {SNRT_COLORS['light_gray']}; padding: 25px; border-radius: 12px; border-left: 5px solid {SNRT_COLORS['primary_blue']};">
                            <h4 style="color: {SNRT_COLORS['primary_blue']}; margin-top: 0; font-size: 18px;">Méthodologie de Validation</h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
                                <div>
                                    <p><strong>🎯 Condition requise:</strong> Au moins 2 pannes historiques par équipement</p>
                                    <p><strong>📊 Justification:</strong> Nécessaire pour établir des patterns de défaillance fiables</p>
                                    <p><strong>🎯 Impact:</strong> Prédictions plus précises sur équipements avec historique suffisant</p>
                                </div>
                                <div>
                                    <p><strong>🔍 Méthodologie:</strong> Analyse temporelle des intervalles entre pannes</p>
                                    <p><strong>⚡ Avantage:</strong> Élimination du bruit et des faux positifs</p>
                                    <p><strong>📈 Résultat:</strong> Modèle prédictif plus robuste et actionnable</p>
                                </div>
                            </div>
                            
                            <div style="background: white; padding: 20px; border-radius: 8px; margin-top: 20px;">
                                <h5 style="color: {SNRT_COLORS['accent_orange']}; margin-top: 0;">Note Importante</h5>
                                <p style="margin: 0; font-style: italic; color: {SNRT_COLORS['dark_gray']};">
                                    Les équipements avec moins de 2 pannes historiques sont automatiquement exclus de cette analyse 
                                    pour garantir la fiabilité des prédictions. Cette approche améliore significativement la précision 
                                    des alertes de maintenance.
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    <div class="page-header">
                        <h2>Liste Détaillée des Équipements Prioritaires</h2>
                        <p>Équipements valides (≥2 pannes) classés par ordre d'urgence croissante</p>
                        <div style="background: {config['couleur_bg']}; padding: 15px; border-radius: 8px; margin-top: 15px;">
                            <small style="color: {SNRT_COLORS['dark_gray']};">
                                <strong>Critère de sélection:</strong> Seuls les équipements avec un historique de pannes suffisant (≥2) 
                                sont inclus pour garantir des prédictions fiables et actionnables.
                            </small>
                        </div>
                    </div>
                    
                    <table>
                        <thead>
                            <tr>
                                <th>Rang</th>
                                <th>Code Équipement</th>
                                <th>Type</th>
                                <th>Localisation</th>
                                <th>Jours avant Panne</th>
                                <th>Date Prévue</th>
                                <th>Niveau de Risque</th>
                                <th>Action Recommandée</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        # Ajouter les lignes du tableau triées par urgence
        for i, pred in enumerate(sorted(predictions_filtrees, key=lambda x: x['days_to_failure']), 1):
            classe_ligne = "priority-critique" if pred['days_to_failure'] <= 7 else ("priority-eleve" if pred['days_to_failure'] <= 30 else "")
            
            # Déterminer la classe CSS pour le niveau de risque
            if pred['risk_level'] == "CRITIQUE":
                classe_risque = "risk-critical"
            elif pred['risk_level'] == "ÉLEVÉ":
                classe_risque = "risk-high"
            elif pred['risk_level'] == "MODÉRÉ":
                classe_risque = "risk-moderate"
            else:
                classe_risque = "risk-low"
            
            contenu_html += f"""
                            <tr class="{classe_ligne}">
                                <td><strong>{i}</strong></td>
                                <td><strong>{pred['equipment_code']}</strong></td>
                                <td>{pred['equipment_type']}</td>
                                <td>{pred['location']}</td>
                                <td><strong>{pred['days_to_failure']}</strong></td>
                                <td>{pred['predicted_failure_date']}</td>
                                <td><span class="{classe_risque}">{pred['risk_level']}</span></td>
                                <td>{pred['recommended_action']}</td>
                            </tr>
            """
        
        contenu_html += f"""
                        </tbody>
                    </table>
                    
                    <div class="actions-section">
                        <h3>Plan d'Action Recommandé</h3>
                        <ul>
        """
        
        # Actions spécifiques selon le type d'alerte
        if type_alerte == 'critique':
            contenu_html += f"""
                            <li><strong>IMMÉDIAT:</strong> Arrêter l'exploitation des {total_equipements} équipement(s) si possible et programmer une intervention immédiate</li>
                            <li><strong>ÉQUIPES:</strong> Mobiliser les équipes de maintenance d'urgence dans les 24h</li>
                            <li><strong>PIÈCES:</strong> Vérifier immédiatement la disponibilité des pièces de rechange critiques</li>
                            <li><strong>COMMUNICATION:</strong> Informer la direction et les équipes d'exploitation sans délai</li>
                            <li><strong>DOCUMENTATION:</strong> Documenter toutes les interventions de manière approfondie</li>
                            <li><strong>SUIVI:</strong> Mettre en place un suivi horaire jusqu'à résolution</li>
            """
        elif type_alerte == 'eleve':
            contenu_html += f"""
                            <li><strong>PLANIFICATION:</strong> Programmer {total_equipements} intervention(s) dans les 2 prochaines semaines</li>
                            <li><strong>RESSOURCES:</strong> Allouer les équipes spécialisées nécessaires</li>
                            <li><strong>INVENTAIRE:</strong> Vérifier et commander les pièces de rechange requises</li>
                            <li><strong>SURVEILLANCE:</strong> Augmenter la fréquence de surveillance à quotidienne</li>
                            <li><strong>COORDINATION:</strong> Coordonner avec les opérations pour les arrêts programmés</li>
                            <li><strong>PRÉPARATION:</strong> Préparer les plans de maintenance préventive détaillés</li>
            """
        elif type_alerte == 'modere':
            contenu_html += f"""
                            <li><strong>INTEGRATION:</strong> Inclure les {total_equipements} équipement(s) dans le planning de maintenance mensuel</li>
                            <li><strong>INSPECTION:</strong> Effectuer des inspections visuelles approfondies</li>
                            <li><strong>MONITORING:</strong> Surveiller les paramètres opérationnels de près</li>
                            <li><strong>DOCUMENTATION:</strong> Mettre à jour la documentation technique</li>
                            <li><strong>ANALYSE:</strong> Examiner l'historique et les tendances de maintenance</li>
                            <li><strong>OPTIMISATION:</strong> Identifier les opportunités d'amélioration</li>
            """
        else:
            contenu_html += f"""
                            <li><strong>SURVEILLANCE:</strong> Continuer les procédures de surveillance normales pour {total_equipements} équipement(s)</li>
                            <li><strong>MAINTENANCE:</strong> Maintenir le programme de maintenance préventive standard</li>
                            <li><strong>MÉTRIQUES:</strong> Documenter régulièrement les métriques de performance</li>
                            <li><strong>TENDANCES:</strong> Examiner les tendances des équipements trimestriellement</li>
                            <li><strong>AMÉLIORATION:</strong> Identifier les bonnes pratiques à généraliser</li>
            """
        
        # Ajouter une section d'insights spécifique au type d'alerte
        if type_alerte == 'critique':
            couleur_insight = SNRT_COLORS['accent_red']
            titre_insight = "Analyse Critique - Action Immédiate Requise"
            insight_message = f"ATTENTION: {total_equipements} équipement(s) valides en situation critique nécessitent une intervention immédiate. Le délai moyen avant panne est de seulement {jours_moyen:.1f} jours."
        elif type_alerte == 'eleve':
            couleur_insight = SNRT_COLORS['accent_orange']
            titre_insight = "Analyse Priorité Élevée - Planification Urgente"
            insight_message = f"PLANIFICATION URGENTE: {total_equipements} équipement(s) valides nécessitent une attention particulière avec un délai moyen de {jours_moyen:.1f} jours."
        else:
            couleur_insight = SNRT_COLORS['primary_blue']
            titre_insight = "Analyse de Maintenance Préventive"
            insight_message = f"MAINTENANCE PRÉVENTIVE: {total_equipements} équipement(s) valides identifiés pour optimisation avec un délai moyen de {jours_moyen:.1f} jours."
        
        contenu_html += f"""
                        </ul>
                    </div>
                    
                    <div style="background: linear-gradient(135deg, {couleur_insight} 0%, {SNRT_COLORS['primary_blue']} 100%); color: white; padding: 30px; border-radius: 12px; margin: 30px 0;">
                        <h3 style="margin-top: 0; font-size: 22px;">{titre_insight}</h3>
                        <p style="margin: 12px 0; font-size: 16px;">{insight_message}</p>
                        <p style="margin: 12px 0; font-size: 16px;"><strong>Recommandation stratégique:</strong> Prioriser les ressources sur les équipements les plus critiques pour maximiser l'impact opérationnel.</p>
                    </div>
                    
                    <div style="background: {SNRT_COLORS['light_gray']}; padding: 30px; border-radius: 12px; margin-top: 40px; text-align: center;">
                        <h3 style="color: {SNRT_COLORS['primary_blue']}; margin-top: 0;">Résumé Exécutif</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                            <div>
                                <p><strong>Niveau de Priorité:</strong> {type_alerte.upper()}</p>
                            </div>
                            <div>
                                <p><strong>Équipements Concernés:</strong> {total_equipements}</p>
                            </div>
                            <div>
                                <p><strong>Délai Moyen:</strong> {jours_moyen:.1f} jours</p>
                            </div>
                            <div>
                                <p><strong>Prochaine Analyse:</strong> {(datetime.now() + timedelta(days=7)).strftime('%d/%m/%Y')}</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    <div class="snrt-colors" style="justify-content: center; margin-bottom: 1rem;">
                        <div class="color-dot color-blue"></div>
                        <div class="color-dot color-red"></div>
                        <div class="color-dot color-orange"></div>
                        <div class="color-dot color-green"></div>
                    </div>
                    <p class="company-name">Société Nationale de Radiodiffusion et de Télévision du Maroc</p>
                    <p><strong>SNRT - DISI</strong></p>
                    <p>Système d'Alertes de Maintenance Prédictive</p>
                    <p>Ce message a été généré automatiquement le {datetime.now().strftime('%d %B %Y à %H:%M')}</p>
                    <p>Pour le support technique, contactez: issami.aymane@gmail.com</p>
                    <p style="margin-top: 15px; font-size: 12px; opacity: 0.8;">
                        Document confidentiel - Distribution limitée au personnel SNRT autorisé
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return contenu_html

    @staticmethod
    def generer_email_rapport_simple(nb_predictions: int, nom_fichier_csv: str) -> str:
        """Générer un email pour le rapport avec fichier CSV utilisant le style SNRT complet"""
        contenu_html = f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport Hebdomadaire de Maintenance SNRT</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: {SNRT_COLORS['light_gray']};
                    line-height: 1.6;
                    color: {SNRT_COLORS['dark_gray']};
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: {SNRT_COLORS['white']};
                    border-radius: 12px;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                    overflow: hidden;
                }}
                .main-header {{
                    background: linear-gradient(135deg, {SNRT_COLORS['primary_blue']} 0%, {SNRT_COLORS['accent_red']} 50%, {SNRT_COLORS['accent_orange']} 100%);
                    padding: 2rem;
                    border-radius: 15px 15px 0 0;
                    color: white;
                    text-align: center;
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                }}
                .main-header h1 {{
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                }}
                .main-header p {{
                    font-size: 1.2rem;
                    opacity: 0.9;
                    margin: 0;
                }}
                .content {{
                    padding: 2rem;
                }}
                .rapport-card {{
                    background: linear-gradient(135deg, {SNRT_COLORS['white']} 0%, {SNRT_COLORS['low_bg']} 100%);
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                    border-left: 5px solid {SNRT_COLORS['primary_blue']};
                    margin: 20px 0;
                }}
                .rapport-title {{
                    font-size: 24px;
                    font-weight: bold;
                    color: {SNRT_COLORS['primary_blue']};
                    margin-bottom: 15px;
                }}
                .rapport-message {{
                    font-size: 16px;
                    color: {SNRT_COLORS['dark_gray']};
                    margin-bottom: 15px;
                }}
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 30px 0;
                }}
                .stat-card {{
                    background: linear-gradient(135deg, {SNRT_COLORS['white']} 0%, {SNRT_COLORS['moderate_bg']} 100%);
                    padding: 25px;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.06);
                    text-align: center;
                    border-left: 4px solid {SNRT_COLORS['accent_orange']};
                }}
                .stat-number {{
                    font-size: 32px;
                    font-weight: bold;
                    color: {SNRT_COLORS['accent_orange']};
                    margin-bottom: 8px;
                }}
                .stat-label {{
                    color: {SNRT_COLORS['dark_gray']};
                    font-size: 14px;
                    font-weight: 600;
                    text-transform: uppercase;
                }}
                .footer {{
                    background-color: {SNRT_COLORS['light_gray']};
                    padding: 2rem;
                    text-align: center;
                    color: {SNRT_COLORS['dark_gray']};
                    font-size: 14px;
                }}
                .snrt-colors {{
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }}
                .color-dot {{
                    width: 12px;
                    height: 12px;
                    border-radius: 50%;
                }}
                .color-blue {{ background-color: {SNRT_COLORS['primary_blue']}; }}
                .color-red {{ background-color: {SNRT_COLORS['accent_red']}; }}
                .color-orange {{ background-color: {SNRT_COLORS['accent_orange']}; }}
                .color-green {{ background-color: {SNRT_COLORS['success_green']}; }}
                .company-name {{
                    font-weight: 600;
                    font-size: 16px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="main-header">
                    <h1>RAPPORT HEBDOMADAIRE DE MAINTENANCE</h1>
                    <p>Système d'Alertes de Maintenance Prédictive</p>
                </div>
                
                <div class="content">
                    <div class="rapport-card">
                        <div class="rapport-title">Rapport de Maintenance - {datetime.now().strftime('%d %B %Y')}</div>
                        <div class="rapport-message">
                            Bonjour,
                        </div>
                        <div class="rapport-message">
                            Veuillez trouver ci-joint le rapport hebdomadaire de maintenance contenant les données d'analyse de tous les équipements.
                        </div>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-number">{nb_predictions}</div>
                            <div class="stat-label">Équipements Analysés</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">CSV</div>
                            <div class="stat-label">Format de Données</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-number">{datetime.now().strftime('%d/%m')}</div>
                            <div class="stat-label">Date de Génération</div>
                        </div>
                    </div>
                    
                    <div style="background: {SNRT_COLORS['light_gray']}; padding: 30px; border-radius: 12px; margin-top: 40px;">
                        <h3 style="color: {SNRT_COLORS['primary_blue']}; margin-top: 0; text-align: center;">Informations du Fichier</h3>
                        <div style="text-align: center;">
                            <p><strong>Nom du fichier:</strong> {nom_fichier_csv}</p>
                            <p><strong>Date de génération:</strong> {datetime.now().strftime('%d/%m/%Y à %H:%M')}</p>
                            <p>Le fichier CSV en pièce jointe contient toutes les informations détaillées sur les prédictions de maintenance pour chaque équipement.</p>
                        </div>
                    </div>
                    
                    <div style="text-align: center; margin-top: 30px; padding: 20px;">
                        <p style="font-size: 16px;">Cordialement,<br>
                        <strong>Système de Maintenance Prédictive SNRT</strong></p>
                    </div>
                </div>
                
                <div class="footer">
                    <div class="snrt-colors" style="justify-content: center; margin-bottom: 1rem;">
                        <div class="color-dot color-blue"></div>
                        <div class="color-dot color-red"></div>
                        <div class="color-dot color-orange"></div>
                        <div class="color-dot color-green"></div>
                    </div>
                    <p class="company-name">Société Nationale de Radiodiffusion et de Télévision du Maroc</p>
                    <p><strong>SNRT - DISI</strong></p>
                    <p>Système d'Alertes de Maintenance Prédictive</p>
                    <p>Ce message a été généré automatiquement le {datetime.now().strftime('%d %B %Y à %H:%M')}</p>
                    <p>Pour le support technique, contactez: issami.aymane@gmail.com</p>
                    <p style="margin-top: 15px; font-size: 12px; opacity: 0.8;">
                        Document confidentiel - Distribution limitée au personnel SNRT autorisé
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return contenu_html

def charger_donnees_equipement(config_donnees: Dict) -> pd.DataFrame:
    """Charger les données d'équipement depuis la source configurée"""
    try:
        if config_donnees['type'] == 'csv':
            return pd.read_csv(config_donnees['chemin'])
        
        elif config_donnees['type'] == 'excel':
            return pd.read_excel(config_donnees['chemin'])
        
        elif config_donnees['type'] == 'database':
            return charger_depuis_bdd(config_donnees)
        
        elif config_donnees['type'] == 'api':
            return charger_depuis_api(config_donnees)
        
        else:
            raise ValueError(f"Type de source de données non supporté: {config_donnees['type']}")
            
    except Exception as e:
        logging.error(f"Erreur de chargement des données: {str(e)}")
        raise

def charger_depuis_bdd(config_bdd: Dict) -> pd.DataFrame:
    """Charger les données depuis la connexion base de données"""
    try:
        import psycopg2
        from sqlalchemy import create_engine
        
        url_connexion = f"postgresql://{config_bdd['username']}:{config_bdd['password']}@{config_bdd['host']}:{config_bdd['port']}/{config_bdd['database']}"
        moteur = create_engine(url_connexion)
        
        requete = f"""
        SELECT 
            code_equipement,
            nom_equipement,
            type_equipement,
            localisation,
            last_maintenance,
            total_pannes,
            total_interventions,
            interventions_7d,
            interventions_30d,
            interventions_90d,
            days_since_last,
            intervention_rate,
            type_encoded,
            DateDebut,
            TypeIntervention,
            equipement
        FROM {config_bdd['table']}
        WHERE code_equipement IS NOT NULL
        ORDER BY code_equipement
        """
        
        df = pd.read_sql(requete, moteur)
        logging.info(f"Données de base de données chargées avec succès: {len(df)} enregistrements")
        return df
        
    except ImportError:
        logging.error("Modules de base de données non installés. Installer avec: pip install psycopg2-binary sqlalchemy")
        raise
    except Exception as e:
        logging.error(f"Erreur de connexion à la base de données: {str(e)}")
        raise

def charger_depuis_api(config_api: Dict) -> pd.DataFrame:
    """Charger les données depuis l'API REST"""
    try:
        import requests
        
        url = f"{config_api['base_url']}/{config_api['endpoint']}"
        headers = config_api.get('headers', {})
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        donnees = response.json()
        
        if isinstance(donnees, list):
            df = pd.DataFrame(donnees)
        elif isinstance(donnees, dict) and 'data' in donnees:
            df = pd.DataFrame(donnees['data'])
        else:
            df = pd.DataFrame([donnees])
        
        logging.info(f"Données API chargées avec succès: {len(df)} enregistrements")
        return df
        
    except ImportError:
        logging.error("Module requests non installé. Installer avec: pip install requests")
        raise
    except Exception as e:
        logging.error(f"Erreur de requête API: {str(e)}")
        raise

def valider_donnees_equipement(df: pd.DataFrame) -> pd.DataFrame:
    """Valider et nettoyer les données d'équipement avec logique Streamlit"""
    colonnes_requises = ['code_equipement']
    
    colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
    if colonnes_manquantes:
        raise ValueError(f"Colonnes requises manquantes: {colonnes_manquantes}")
    
    df_propre = df.copy()
    df_propre = df_propre.dropna(subset=['code_equipement'])
    
    # ÉTAPE 1: Conversion de la colonne date (CRUCIAL pour la logique temporelle)
    if 'DateDebut' in df_propre.columns:
        df_propre['DateDebut'] = pd.to_datetime(df_propre['DateDebut'], errors='coerce')
    else:
        if 'last_maintenance' in df_propre.columns:
            df_propre['DateDebut'] = pd.to_datetime(df_propre['last_maintenance'], errors='coerce')
        else:
            # Créer des dates historiques synthétiques
            df_propre['DateDebut'] = pd.date_range(
                start=datetime.now() - timedelta(days=365),
                end=datetime.now(),
                periods=len(df_propre)
            )
    
    # ÉTAPE 2: Supprimer les lignes sans date et trier (NOUVEAU - CRUCIAL)
    df_propre = df_propre.dropna(subset=['DateDebut']).copy()
    df_propre = df_propre.sort_values(['code_equipement', 'DateDebut'])
    
    # ÉTAPE 3: Identifier les pannes (CRUCIAL pour le filtrage)
    if 'is_panne' not in df_propre.columns:
        if 'TypeIntervention' in df_propre.columns:
            df_propre['is_panne'] = (df_propre['TypeIntervention'] == 'incident').astype(int)
        else:
            df_propre['is_panne'] = 0
    
    # Valeurs par défaut compatibles
    valeurs_par_defaut = {
        'nom_equipement': 'Inconnu',
        'type_equipement': 'Inconnu',
        'equipement': 'Inconnu',
        'localisation': 'Inconnu',
        'total_pannes': 0,
        'total_interventions': 0,
        'interventions_7d': 0,
        'interventions_30d': 0,
        'interventions_90d': 0,
        'days_since_last': 30,
        'intervention_rate': 0.1,
        'type_encoded': 0,
        'TypeIntervention': 'maintenance'
    }
    
    for colonne, valeur_defaut in valeurs_par_defaut.items():
        if colonne not in df_propre.columns:
            df_propre[colonne] = valeur_defaut
        else:
            df_propre[colonne] = df_propre[colonne].fillna(valeur_defaut)
    
    # Traitement des colonnes numériques
    colonnes_numeriques = ['total_pannes', 'total_interventions', 'interventions_7d', 
                          'interventions_30d', 'interventions_90d', 'days_since_last', 
                          'intervention_rate', 'type_encoded']
    
    for col in colonnes_numeriques:
        if col in df_propre.columns:
            df_propre[col] = pd.to_numeric(df_propre[col], errors='coerce').fillna(0)
    
    # S'assurer qu'il n'y a pas de dates nulles
    date_par_defaut = datetime.now() - timedelta(days=30)
    df_propre['DateDebut'] = df_propre['DateDebut'].fillna(date_par_defaut)
    
    # Synchroniser les colonnes de nom d'équipement
    if 'nom_equipement' in df_propre.columns and 'equipement' not in df_propre.columns:
        df_propre['equipement'] = df_propre['nom_equipement']
    elif 'equipement' in df_propre.columns and 'nom_equipement' not in df_propre.columns:
        df_propre['nom_equipement'] = df_propre['equipement']
    
    # Synchroniser les colonnes de type d'équipement
    if 'type_equipement' not in df_propre.columns and 'equipement' in df_propre.columns:
        df_propre['type_equipement'] = df_propre['equipement']
    
    logging.info(f"Validation des données terminée: {len(df_propre)} enregistrements traités")
    logging.info(f"Colonnes disponibles: {list(df_propre.columns)}")
    logging.info(f"Période des données: {df_propre['DateDebut'].min()} à {df_propre['DateDebut'].max()}")
    
    # NOUVEAU: Statistiques sur les pannes
    total_pannes = df_propre['is_panne'].sum()
    taux_pannes = df_propre['is_panne'].mean() * 100
    logging.info(f"Statistiques pannes: {total_pannes} pannes sur {len(df_propre)} interventions ({taux_pannes:.1f}%)")
    
    return df_propre

def generer_rapport_csv(predictions: List[Dict]) -> List[str]:
    """Générer un rapport CSV détaillé des prédictions avec la structure demandée"""
    try:
        nom_fichier = f"output/predictions_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Extraire seulement les colonnes demandées
        donnees_csv = []
        for pred in predictions:
            donnees_csv.append({
                'equipment_code': pred.get('equipment_code', ''),
                'current_date': pred.get('current_date', ''),
                'predicted_failure_date': pred.get('predicted_failure_date', ''),
                'days_to_failure': pred.get('days_to_failure', 0),
                'risk_level': pred.get('risk_level', ''),
                'recommended_action': pred.get('recommended_action', ''),
            })
        
        df = pd.DataFrame(donnees_csv)
        df.to_csv(nom_fichier, index=False, encoding='utf-8-sig')
        
        logging.info(f"Rapport CSV généré: {nom_fichier}")
        return [nom_fichier]
    except Exception as e:
        logging.error(f"Erreur de génération CSV: {str(e)}")
        return []

def sauvegarder_historique_predictions(predictions: List[Dict]):
    """Sauvegarder l'historique détaillé des prédictions avec métadonnées analytiques"""
    try:
        fichier_historique = "history.json"
        
        if os.path.exists(fichier_historique):
            with open(fichier_historique, 'r', encoding='utf-8') as f:
                historique = json.load(f)
        else:
            historique = []
        
        # Calculer des statistiques complètes
        tous_jours = [p['days_to_failure'] for p in predictions if 'days_to_failure' in p]
        
        entree_historique = {
            'timestamp': datetime.now().isoformat(),
            'total_predictions': len(predictions),
            'critique_count': len([p for p in predictions if p['priority'] == 'critique']),
            'eleve_count': len([p for p in predictions if p['priority'] == 'eleve']),
            'modere_count': len([p for p in predictions if p['priority'] == 'modere']),
            'faible_count': len([p for p in predictions if p['priority'] == 'faible']),
            'statistics': {
                'avg_days_to_failure': float(np.mean(tous_jours)) if tous_jours else 0,
                'median_days_to_failure': float(np.median(tous_jours)) if tous_jours else 0,
                'std_days_to_failure': float(np.std(tous_jours)) if tous_jours else 0,
                'min_days_to_failure': int(min(tous_jours)) if tous_jours else 0,
                'max_days_to_failure': int(max(tous_jours)) if tous_jours else 0
            },
            'urgent_equipment_count': len([p for p in predictions if p['days_to_failure'] <= 30]),
            'equipment_types': len(set([p.get('equipment_type', 'Inconnu') for p in predictions])),
            'predictions': predictions
        }
        
        historique.append(entree_historique)
        
        # Maintenir un historique de 90 jours
        date_limite = datetime.now() - timedelta(days=90)
        historique = [h for h in historique if datetime.fromisoformat(h['timestamp']) > date_limite]
        
        with open(fichier_historique, 'w', encoding='utf-8') as f:
            json.dump(historique, f, ensure_ascii=False, indent=2, default=str)
        
        logging.info(f"Historique des prédictions sauvegardé: {len(historique)} entrées maintenues")
        
    except Exception as e:
        logging.error(f"Erreur de sauvegarde de l'historique: {str(e)}")

def run_scheduler():
    """Exécute le planificateur selon la configuration"""
    try:
        # Charger la configuration
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        planificateur_config = config.get('planificateur', {})
        if not planificateur_config.get('active', False):
            print("Planificateur désactivé dans la configuration")
            return
        
        # Configuration du planificateur
        timezone = pytz.timezone(planificateur_config.get('timezone', 'Africa/Casablanca'))
        scheduler = BlockingScheduler(timezone=timezone)
        
        # Configuration mensuelle
        if planificateur_config.get('type') == 'monthly':
            jour_du_mois = planificateur_config.get('jour_du_mois', 1)
            heure_minute = planificateur_config.get('heure', '09:00').split(':')
            heure = int(heure_minute[0])
            minute = int(heure_minute[1])
            
            scheduler.add_job(
                main,
                CronTrigger(day=jour_du_mois, hour=heure, minute=minute),
                id='maintenance_mensuelle'
            )
            
            print(f"Planificateur configuré pour s'exécuter le {jour_du_mois} de chaque mois à {heure:02d}:{minute:02d}")
            print(f"Timezone: {timezone}")
            print("Planificateur en cours d'exécution... (Ctrl+C pour arrêter)")
        
        # Configuration hebdomadaire
        elif planificateur_config.get('type') == 'weekly':
            jour_de_la_semaine = planificateur_config.get('jour_de_la_semaine', 1)  # 1 = Lundi
            heure_minute = planificateur_config.get('heure', '09:00').split(':')
            heure = int(heure_minute[0])
            minute = int(heure_minute[1])
            
            scheduler.add_job(
                main,
                CronTrigger(day_of_week=jour_de_la_semaine, hour=heure, minute=minute),
                id='maintenance_hebdomadaire'
            )
            
            jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
            jour_nom = jours_semaine[jour_de_la_semaine] if 0 <= jour_de_la_semaine <= 6 else 'Lundi'
            print(f"Planificateur configuré pour s'exécuter chaque {jour_nom} à {heure:02d}:{minute:02d}")
            print(f"Timezone: {timezone}")
            print("Planificateur en cours d'exécution... (Ctrl+C pour arrêter)")
        
        scheduler.start()
        
    except Exception as e:
        logging.error(f"Erreur du planificateur: {str(e)}")
        print(f"Erreur du planificateur: {str(e)}")

def main():
    """Fonction d'exécution principale"""
    logger = logging.getLogger(__name__)
    logger.info("Initialisation du Système d'Alertes de Maintenance Prédictive")
    
    try:
        # Gestion de la configuration
        fichier_config = 'config.json'
        
        if not os.path.exists(fichier_config):
            print("\n" + "="*80)
            print("SYSTÈME DE MAINTENANCE PRÉDICTIVE SNRT - CONFIGURATION INITIALE")
            print("="*80)
            print("Système professionnel de prédiction de maintenance")
            
            
            
            chemin_modele = obtenir_chemin_modele()
            config_donnees = obtenir_source_donnees()
            
            # Validation de la configuration
            print("\nValidation de la configuration...")
            try:
                donnees_test = charger_donnees_equipement(config_donnees)
                donnees_test = valider_donnees_equipement(donnees_test)
                print(f"Validation des données réussie: {len(donnees_test)} enregistrements chargés")
                
                predicteur_test = PredicteurMaintenance(chemin_modele, donnees_test)
                print("Chargement du modèle réussi")
                
            except Exception as e:
                print(f"Échec de la validation de la configuration: {str(e)}")
                print("Veuillez vérifier vos paramètres et redémarrer le système")
                return
            
            config = {
                'chemin_modele': chemin_modele,
                'config_donnees': config_donnees,
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'info_systeme': {
                    'organisation': 'SNRT - Société Nationale de Radiodiffusion et de Télévision du Maroc',
                    'auteur': 'Aymane ISSAMI',
                    'description': 'Système professionnel d\'alertes de maintenance prédictive'
                }
            }
            
            with open(fichier_config, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\nConfiguration sauvegardée avec succès: {fichier_config}")
            print("La configuration peut être modifiée en éditant directement ce fichier")
        
        else:
            with open(fichier_config, 'r') as f:
                config = json.load(f)
            
            chemin_modele = config['chemin_modele']
            config_donnees = config['config_donnees']
            
            logger.info(f"Configuration chargée depuis: {fichier_config}")
            if config.get('version') != '1.0':
                logger.info("Système mis à jour v1.0")
        
        # Vérification de la disponibilité du modèle
        if not os.path.exists(chemin_modele):
            logger.error(f"Fichier de modèle non trouvé: {chemin_modele}")
            print(f"\nErreur: Le fichier de modèle n'existe pas: {chemin_modele}")
            print("Supprimez config.json pour reconfigurer le système")
            return
        
        # Initialisation des services
        service_email = ServiceEmailMaintenance(SMTP_CONFIG)
        generateur_template = GenerateurTemplateEmail()
        
        # Chargement et traitement des données d'équipement
        logger.info("Chargement des données d'équipement pour l'analyse...")
        donnees_equipement = charger_donnees_equipement(config_donnees)
        donnees_equipement = valider_donnees_equipement(donnees_equipement)
        
        logger.info(f"Données d'équipement traitées: {len(donnees_equipement)} enregistrements prêts pour l'analyse")
        
        # Chargement du modèle et exécution des prédictions
        logger.info(f"Initialisation du modèle de prédiction: {chemin_modele}")
        predicteur = PredicteurMaintenance(chemin_modele, donnees_equipement)
        
        # Exécuter les prédictions pour tous les équipements uniques
        equipements_uniques = donnees_equipement['code_equipement'].nunique()
        logger.info(f"Exécution des prédictions pour {equipements_uniques} équipements uniques")
        
        predictions = predicteur.predire_panne_equipement(donnees_equipement)
        
        logger.info(f"Analyse de prédiction terminée: {len(predictions)} équipements traités")
        
        # Résumé statistique et analyse avec validation
        if predictions:
            # Afficher les statistiques de validité
            total_equipements = donnees_equipement['code_equipement'].nunique()
            equipements_valides = len(predicteur.valid_equipments) if predicteur.valid_equipments else 0
            equipements_exclus = total_equipements - equipements_valides
            
            logger.info(f"STATISTIQUES DE VALIDITÉ:")
            logger.info(f"  - Total équipements: {total_equipements}")
            logger.info(f"  - Équipements valides (≥2 pannes): {equipements_valides}")
            logger.info(f"  - Équipements exclus (<2 pannes): {equipements_exclus}")
            if total_equipements > 0:
                logger.info(f"  - Taux de validité: {equipements_valides/total_equipements*100:.1f}%")
            
            compte_critique = len([p for p in predictions if p['priority'] == 'critique'])
            compte_eleve = len([p for p in predictions if p['priority'] == 'eleve'])
            compte_modere = len([p for p in predictions if p['priority'] == 'modere'])
            compte_faible = len([p for p in predictions if p['priority'] == 'faible'])
            
            print(f"\nRÉSUMÉ DE L'ANALYSE DES PRÉDICTIONS:")
            print(f"   Total équipements dans les données: {total_equipements}")
            print(f"   Équipements valides (≥2 pannes): {equipements_valides}")
            print(f"   Équipements exclus (<2 pannes): {equipements_exclus}")
            if total_equipements > 0:
                print(f"   Taux de validité: {equipements_valides/total_equipements*100:.1f}%")
            print(f"   ────────────────────────────────────────")
            print(f"   Équipements analysés: {len(predictions)} sur {equipements_valides} valides")
            print(f"   Priorité CRITIQUE: {compte_critique} équipements (≤ 7 jours)")
            print(f"   Priorité ÉLEVÉE: {compte_eleve} équipements (8-30 jours)")
            print(f"   Priorité MODÉRÉE: {compte_modere} équipements (31-90 jours)")
            print(f"   Priorité FAIBLE: {compte_faible} équipements (> 90 jours)")
            
            # Analyse statistique avancée
            tous_jours = [p['days_to_failure'] for p in predictions]
            if tous_jours:
                jours_moyens = np.mean(tous_jours)
                jours_min = min(tous_jours)
                jours_max = max(tous_jours)
                jours_mediane = np.median(tous_jours)
                print(f"\nANALYSE STATISTIQUE (Équipements Valides):")
                print(f"   Temps moyen avant panne: {jours_moyens:.1f} jours")
                print(f"   Temps médian avant panne: {jours_mediane:.1f} jours")
                print(f"   Équipement le plus urgent: {jours_min} jour(s)")
                print(f"   Équipement le moins urgent: {jours_max} jour(s)")
                
            # Identification des équipements prioritaires
            priorite_haute = sorted([p for p in predictions if p['days_to_failure'] <= 30], 
                                  key=lambda x: x['days_to_failure'])[:10]
            if priorite_haute:
                print(f"\nÉQUIPEMENTS PRIORITAIRES (Valides):")
                for i, pred in enumerate(priorite_haute, 1):
                    print(f"   {i}. {pred['equipment_code']} - {pred['days_to_failure']} jour(s) ({pred['risk_level']})")
            
            # Avertissement si beaucoup d'équipements sont exclus
            if equipements_exclus > equipements_valides:
                print(f"\nATTENTION: {equipements_exclus} équipements exclus (données insuffisantes)")
                print(f"   Pour améliorer la couverture, considérez:")
                print(f"   - Enrichir l'historique des pannes")
                print(f"   - Ajuster le critère de validité (actuellement ≥2 pannes)")
                print(f"   - Vérifier la qualité des données TypeIntervention")
            
            # Recommandations d'optimisation du système
            total_urgent = compte_critique + compte_eleve
            if len(predictions) > 0 and total_urgent > len(predictions) * 0.25:
                print(f"\nRECOMMANDATION SYSTÈME: Demande de maintenance élevée détectée ({total_urgent/len(predictions)*100:.1f}%)")
                print("   Considérer l'augmentation de la fréquence de maintenance préventive")
                print("   Examiner la capacité et l'allocation des ressources de l'équipe de maintenance")
                print("   Analyser les modèles de panne d'équipement par type et localisation")
            
            # Recommandations pour améliorer la couverture
            if total_equipements > 0:
                couverture = len(predictions) / total_equipements * 100
                if couverture < 50:
                    print(f"\nRECOMMANDATION DONNÉES: Couverture prédictive faible ({couverture:.1f}%)")
                    print("   - Enrichir l'historique des incidents pour plus d'équipements")
                    print("   - Standardiser la classification des types d'intervention")
                    print("   - Considérer un modèle alternatif pour équipements avec peu d'historique")
            
        else:
            print("\n❌ AUCUNE PRÉDICTION GÉNÉRÉE")
            if hasattr(predicteur, 'valid_equipments') and predicteur.valid_equipments is not None:
                print(f"   Équipements valides disponibles: {len(predicteur.valid_equipments)}")
            else:
                print("   Aucun équipement valide trouvé (critère: ≥2 pannes)")
            print("   Vérifiez la qualité des données et la compatibilité du modèle")
        
        # Groupement des alertes basé sur les risques
        predictions_groupees = {
            'critique': [p for p in predictions if p['priority'] == 'critique'],
            'eleve': [p for p in predictions if p['priority'] == 'eleve'],
            'modere': [p for p in predictions if p['priority'] == 'modere'],
            'faible': [p for p in predictions if p['priority'] == 'faible']
        }
        
        # Système d'alertes email professionnel - ENVOI SYSTÉMATIQUE DE 5 EMAILS
        emails_envoyes = []
        
        # PRIORITÉS - Toujours envoyer les 4 emails de priorité même s'il n'y a pas d'équipements
        priorites = ['critique', 'eleve', 'modere', 'faible']
        
        for priorite in priorites:
            nb_equipements = len(predictions_groupees[priorite])
            logger.info(f"Envoi d'alertes de priorité {priorite} pour {nb_equipements} équipements")
            
            # Générer le contenu même s'il n'y a pas d'équipements de cette priorité
            if nb_equipements > 0:
                contenu_html = generateur_template.generer_email_alerte(predictions, priorite)
            else:
                # Créer un email avec le même style SNRT indiquant qu'il n'y a pas d'équipements de cette priorité
                configs_alerte = {
                    'critique': {'titre': 'ALERTE CRITIQUE - Aucun Équipement', 'couleur': SNRT_COLORS['accent_red'], 'couleur_bg': SNRT_COLORS['critical_bg']},
                    'eleve': {'titre': 'ALERTE PRIORITÉ ÉLEVÉE - Aucun Équipement', 'couleur': SNRT_COLORS['accent_orange'], 'couleur_bg': SNRT_COLORS['high_bg']},
                    'modere': {'titre': 'ALERTE MODÉRÉE - Aucun Équipement', 'couleur': SNRT_COLORS['warning_yellow'], 'couleur_bg': SNRT_COLORS['moderate_bg']},
                    'faible': {'titre': 'INFORMATION - Aucun Équipement', 'couleur': SNRT_COLORS['success_green'], 'couleur_bg': SNRT_COLORS['low_bg']}
                }
                config = configs_alerte[priorite]
                
                contenu_html = f"""
                <!DOCTYPE html>
                <html lang="fr">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>{config['titre']}</title>
                    <style>
                        body {{
                            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                            margin: 0;
                            padding: 20px;
                            background-color: {SNRT_COLORS['light_gray']};
                            line-height: 1.6;
                            color: {SNRT_COLORS['dark_gray']};
                        }}
                        .container {{
                            max-width: 1200px;
                            margin: 0 auto;
                            background-color: {SNRT_COLORS['white']};
                            border-radius: 12px;
                            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                            overflow: hidden;
                        }}
                        .main-header {{
                            background: linear-gradient(135deg, {SNRT_COLORS['primary_blue']} 0%, {SNRT_COLORS['accent_red']} 50%, {SNRT_COLORS['accent_orange']} 100%);
                            padding: 2rem;
                            border-radius: 15px 15px 0 0;
                            color: white;
                            text-align: center;
                            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                        }}
                        .main-header h1 {{
                            font-size: 2.5rem;
                            font-weight: 700;
                            margin-bottom: 0.5rem;
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                        }}
                        .main-header p {{
                            font-size: 1.2rem;
                            opacity: 0.9;
                            margin: 0;
                        }}
                        .content {{
                            padding: 2rem;
                        }}
                        .status-card {{
                            background: linear-gradient(135deg, {SNRT_COLORS['white']} 0%, {config['couleur_bg']} 100%);
                            padding: 30px;
                            border-radius: 12px;
                            box-shadow: 0 6px 20px rgba(0,0,0,0.08);
                            text-align: center;
                            border-left: 5px solid {config['couleur']};
                            margin: 20px 0;
                        }}
                        .status-title {{
                            font-size: 24px;
                            font-weight: bold;
                            color: {config['couleur']};
                            margin-bottom: 15px;
                        }}
                        .status-message {{
                            font-size: 16px;
                            color: {SNRT_COLORS['dark_gray']};
                            margin-bottom: 10px;
                        }}
                        .footer {{
                            background-color: {SNRT_COLORS['light_gray']};
                            padding: 2rem;
                            text-align: center;
                            color: {SNRT_COLORS['dark_gray']};
                            font-size: 14px;
                        }}
                        .snrt-colors {{
                            display: flex;
                            align-items: center;
                            gap: 0.5rem;
                        }}
                        .color-dot {{
                            width: 12px;
                            height: 12px;
                            border-radius: 50%;
                        }}
                        .color-blue {{ background-color: {SNRT_COLORS['primary_blue']}; }}
                        .color-red {{ background-color: {SNRT_COLORS['accent_red']}; }}
                        .color-orange {{ background-color: {SNRT_COLORS['accent_orange']}; }}
                        .color-green {{ background-color: {SNRT_COLORS['success_green']}; }}
                        .company-name {{
                            font-weight: 600;
                            font-size: 16px;
                        }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="main-header">
                            <h1>{config['titre']}</h1>
                            <p>Système d'Alertes de Maintenance Prédictive</p>
                        </div>
                        
                        <div class="content">
                            <div class="status-card">
                                <div class="status-title">État: Aucun équipement de priorité {priorite.upper()}</div>
                                <div class="status-message">
                                    Aucun équipement ne nécessite d'intervention de priorité {priorite} à ce jour.
                                </div>
                                <div class="status-message">
                                    <strong>Date de vérification:</strong> {datetime.now().strftime('%d %B %Y à %H:%M')}
                                </div>
                            </div>
                            
                            <div style="background: {SNRT_COLORS['light_gray']}; padding: 30px; border-radius: 12px; margin-top: 40px; text-align: center;">
                                <h3 style="color: {SNRT_COLORS['primary_blue']}; margin-top: 0;">Résumé Exécutif</h3>
                                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0;">
                                    <div>
                                        <p><strong>Niveau de Priorité:</strong> {priorite.upper()}</p>
                                    </div>
                                    <div>
                                        <p><strong>Équipements Concernés:</strong> 0</p>
                                    </div>
                                    <div>
                                        <p><strong>Prochaine Analyse:</strong> {(datetime.now() + timedelta(days=7)).strftime('%d/%m/%Y')}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="footer">
                            <div class="snrt-colors" style="justify-content: center; margin-bottom: 1rem;">
                                <div class="color-dot color-blue"></div>
                                <div class="color-dot color-red"></div>
                                <div class="color-dot color-orange"></div>
                                <div class="color-dot color-green"></div>
                            </div>
                            <p class="company-name">Société Nationale de Radiodiffusion et de Télévision du Maroc</p>
                            <p><strong>SNRT - DISI</strong></p>
                            <p>Système d'Alertes de Maintenance Prédictive</p>
                            <p>Ce message a été généré automatiquement le {datetime.now().strftime('%d %B %Y à %H:%M')}</p>
                            <p>Pour le support technique, contactez: issami.aymane@gmail.com</p>
                            <p style="margin-top: 15px; font-size: 12px; opacity: 0.8;">
                                Document confidentiel - Distribution limitée au personnel SNRT autorisé
                            </p>
                        </div>
                    </div>
                </body>
                </html>
                """
            
            if contenu_html:
                # Définir les destinataires selon la priorité
                if priorite in RECIPIENTS:
                    destinataires = RECIPIENTS[priorite]
                else:
                    destinataires = RECIPIENTS.get('rapports', ['issami.aymane@gmail.com'])
                
                succes = service_email.envoyer_email(
                    emails_destinataires=destinataires,
                    sujet=f"ALERTE {priorite.upper()} MAINTENANCE SNRT - {nb_equipements} Équipement(s) - {datetime.now().strftime('%d/%m/%Y')}",
                    contenu_html=contenu_html
                )
                if succes:
                    logger.info(f"Alerte de priorité {priorite} envoyée avec succès")
                    emails_envoyes.append(f"Alerte {priorite.capitalize()}")
                else:
                    logger.error(f"Échec de l'envoi de l'alerte de priorité {priorite}")
        
        
        # 5ÈME EMAIL - RAPPORT SIMPLE AVEC CSV - TOUJOURS ENVOYÉ
        logger.info("Génération du rapport simple avec fichier CSV")
        pieces_jointes_csv = generer_rapport_csv(predictions) if predictions else None
        
        # Créer un email simple avec le fichier CSV
        nb_predictions = len(predictions) if predictions else 0
        nom_fichier = pieces_jointes_csv[0] if pieces_jointes_csv else "Aucun fichier généré"
        
        contenu_rapport_simple = generateur_template.generer_email_rapport_simple(nb_predictions, nom_fichier)
        
        succes = service_email.envoyer_email(
            emails_destinataires=RECIPIENTS['rapports'],
            sujet=f"Rapport Hebdomadaire de Maintenance SNRT - {datetime.now().strftime('%d %B %Y')}",
            contenu_html=contenu_rapport_simple,
            pieces_jointes=pieces_jointes_csv if pieces_jointes_csv else None
        )
        if succes:
            logger.info("Rapport hebdomadaire simple envoyé avec succès")
            emails_envoyes.append("Rapport Hebdomadaire Simple")
        else:
            logger.error("Échec de l'envoi du rapport hebdomadaire simple")

        # Préservation des données historiques et analytics
        if predictions:  # Sauvegarder seulement s'il y a des prédictions
            sauvegarder_historique_predictions(predictions)
        
        # Résumé d'exécution et rapport de statut
        logger.info("Exécution du système de prédiction de maintenance terminée avec succès")
        
        if emails_envoyes:
            print(f"\nNOTIFICATIONS EMAIL ENVOYÉES: {', '.join(emails_envoyes)} ({len(emails_envoyes)} emails au total)")
        else:
            print(f"\nAucune notification email envoyée - Erreur système")
            
        # Recommandations opérationnelles avec vérifications
        if predictions:  # S'assurer qu'il y a des prédictions
            if compte_critique > 0:
                print(f"\nACTION IMMÉDIATE REQUISE: {compte_critique} équipement(s) valides nécessitent une intervention critique")
            elif compte_eleve > 8:
                print(f"\nATTENTION REQUISE: {compte_eleve} équipements valides nécessitent une planification urgente")
            else:
                print(f"\nSTATUS SYSTÈME: Condition opérationnelle stable sur les équipements valides")
        else:
            print(f"\nSTATUS SYSTÈME: Aucune prédiction générée - vérifier les données d'entrée")
        
        # Recommandations d'optimisation du système avec vérifications
        if predictions:
            total_urgent = compte_critique + compte_eleve
            if total_urgent > len(predictions) * 0.25:
                print(f"\nRECOMMANDATION SYSTÈME: Demande de maintenance élevée détectée ({total_urgent/len(predictions)*100:.1f}%)")
                print("   Considérer l'augmentation de la fréquence de maintenance préventive")
                print("   Examiner la capacité et l'allocation des ressources de l'équipe de maintenance")
                print("   Analyser les modèles de panne d'équipement par type et localisation")
        
        logger.info(f"Session terminée avec succès - {len(predictions)} prédictions générées, {len(emails_envoyes)} notifications envoyées")
        
    except KeyboardInterrupt:
        logger.info("Exécution du système interrompue par l'utilisateur")
        print("\nExécution du système arrêtée sur demande de l'utilisateur")
        
    except Exception as e:
        logger.error(f"Erreur critique du système: {str(e)}")
        print(f"\nERREUR CRITIQUE DU SYSTÈME: {str(e)}")
        
        # Notification d'erreur automatisée aux administrateurs système
        try:
            service_email = ServiceEmailMaintenance(SMTP_CONFIG)
            horodatage_erreur = datetime.now().strftime('%d %B %Y à %H:%M:%S')
            
            email_erreur = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Notification d'Erreur Système SNRT</title>
            </head>
            <body style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: {SNRT_COLORS['dark_gray']}; line-height: 1.6;">
                <div style="max-width: 700px; margin: 0 auto; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.1);">
                    <div style="background: linear-gradient(135deg, {SNRT_COLORS['accent_red']} 0%, {SNRT_COLORS['dark_gray']} 100%); color: white; padding: 40px; text-align: center;">
                        <h1 style="margin: 0; font-size: 28px; font-weight: 700;">Notification d'Erreur Système SNRT</h1>
                        <p style="margin: 15px 0 0 0; font-size: 16px; opacity: 0.95;">Intervention technique requise</p>
                    </div>
                    <div style="padding: 40px;">
                        <h3 style="color: {SNRT_COLORS['accent_red']}; margin-top: 0; font-size: 20px;">Détails de l'Erreur</h3>
                        <table style="width: 100%; border-collapse: collapse; margin: 25px 0;">
                            <tr style="border-bottom: 1px solid {SNRT_COLORS['light_gray']};">
                                <td style="padding: 12px 0; font-weight: 600; width: 30%;">Horodatage:</td>
                                <td style="padding: 12px 0;">{horodatage_erreur}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid {SNRT_COLORS['light_gray']};">
                                <td style="padding: 12px 0; font-weight: 600;">Version du Système:</td>
                                <td style="padding: 12px 0;">v1.0</td>
                            </tr>
                            <tr style="border-bottom: 1px solid {SNRT_COLORS['light_gray']};">
                                <td style="padding: 12px 0; font-weight: 600;">Organisation:</td>
                                <td style="padding: 12px 0;">SNRT - Société Nationale de Radiodiffusion et de Télévision</td>
                            </tr>
                            <tr style="border-bottom: 1px solid {SNRT_COLORS['light_gray']};">
                                <td style="padding: 12px 0; font-weight: 600;">Description de l'Erreur:</td>
                                <td style="padding: 12px 0; color: {SNRT_COLORS['accent_red']}; font-family: monospace; font-size: 14px;">{str(e)[:300]}...</td>
                            </tr>
                        </table>
                        
                        <div style="margin: 35px 0; padding: 25px; background: {SNRT_COLORS['critical_bg']}; border-radius: 12px; border-left: 5px solid {SNRT_COLORS['accent_red']};">
                            <h4 style="margin-top: 0; color: {SNRT_COLORS['accent_red']}; font-size: 18px;">Actions Requises</h4>
                            <ul style="margin-bottom: 0; padding-left: 20px;">
                                <li style="margin: 8px 0;">Examiner les journaux système détaillés: alerts.log</li>
                                <li style="margin: 8px 0;">Vérifier la disponibilité du fichier de modèle et la connectivité de la source de données</li>
                                <li style="margin: 8px 0;">Tester la connectivité réseau et la fonctionnalité du service email</li>
                                <li style="margin: 8px 0;">Redémarrer le système après résolution des problèmes identifiés</li>
                                <li style="margin: 8px 0;">Contacter le support technique si l'erreur persiste</li>
                            </ul>
                        </div>
                        
                        <div style="text-align: center; margin-top: 35px; padding-top: 25px; border-top: 2px solid {SNRT_COLORS['light_gray']};">
                            <p style="color: {SNRT_COLORS['primary_blue']}; font-size: 16px; font-weight: 600; margin: 0;">
                                Système de Maintenance Prédictive SNRT
                            </p>
                            <p style="color: {SNRT_COLORS['dark_gray']}; font-size: 14px; margin: 5px 0 0 0;">
                                Support Technique: issami.aymane@gmail.com
                            </p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            service_email.envoyer_email(
                emails_destinataires=['issami.aymane@gmail.com'],
                sujet=f"Erreur Système SNRT - {horodatage_erreur}",
                contenu_html=email_erreur
            )
            logger.info("Notification d'erreur envoyée aux administrateurs système")
            
        except Exception as erreur_email:
            logger.error(f"Échec de l'envoi de la notification d'erreur: {str(erreur_email)}")
        
        print("\nPour le support technique, contactez: issami.aymane@gmail.com")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Système d'Alertes de Maintenance Prédictive visant à prédire le moment probable de défaillance d’un équipement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Exemples d'Utilisation:
            python3 alerts.py --mode once        # Mode d'exécution unique avec configuration interactive
            python3 alerts.py --mode schedule    # Mode de planification automatisée
            python3 alerts.py --config           # Reconfiguration du système
            python3 alerts.py --test             # Test de configuration sans envoi d'emails
        """
    )
    
    parser.add_argument("--mode", choices=["once", "schedule"], default="once",
                       help="Mode d'exécution: exécution unique ou planification automatisée")
    parser.add_argument("--config", action="store_true",
                       help="Reconfigurer les paramètres système (modèle et source de données)")
    parser.add_argument("--test", action="store_true",
                       help="Tester la configuration actuelle sans envoyer d'emails")
    
    args = parser.parse_args()
    
    
    # Gestion des opérations spéciales
    if args.config:
        fichier_config = 'config.json'
        if os.path.exists(fichier_config):
            fichier_sauvegarde = f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.rename(fichier_config, fichier_sauvegarde)
            print(f"Configuration existante sauvegardée: {fichier_sauvegarde}")
        else:
            print("Aucune configuration existante trouvée.")
        print("Redémarrez le système pour commencer le processus de configuration")
        exit(0)
    
    if args.test:
        print("MODE TEST - Les notifications email seront supprimées")
        # Modifier la configuration SMTP pour les tests
        SMTP_CONFIG['test_mode'] = True
        # Rediriger les destinataires d'email pour les tests
        RECIPIENTS = {cle: ['test@example.com'] for cle in RECIPIENTS.keys()}
    
    # Affichage d'initialisation du système
    print("\n" + "="*80)
    print("Système d'Alertes de Maintenance Prédictive")
    print("="*80)
    print(f"Heure d'Exécution: {datetime.now().strftime('%d %B %Y à %H:%M:%S')}")
    print(f"Mode d'Opération: {args.mode.upper()}")
    if args.test:
        print("MODE TEST ACTIF - Aucune notification email réelle ne sera envoyée")
    print("="*80)
    
    # Exécution principale du système
    if args.mode == "once":
        logging.info("Mode d'exécution unique - Système d'Alertes de Maintenance Prédictive")
        main()
    else:
        logging.info("Mode de planification automatisée - Système d'Alertes de Maintenance Prédictive")
        try:
            run_scheduler()
        except KeyboardInterrupt:
            print("\nPlanification automatisée arrêtée par l'utilisateur")
            logging.info("Planification automatisée terminée")