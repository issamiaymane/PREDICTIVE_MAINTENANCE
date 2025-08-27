#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classe PredictionDashboard pour la gestion des prédictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
from config.settings import RISK_LEVELS, DEFAULT_PROBABILITIES, CONFIDENCE_INTERVAL_FACTOR, ANALYSIS_LIMITS

class PredictionDashboard:
    """
    Classe principale pour la gestion des prédictions de maintenance
    """
    
    def __init__(self):
        self.model_package = None
        self.df_clean = None
        self.model_df = None  # Dataset enrichi avec features temporelles
        self.valid_equipments = None  # Équipements avec au moins 2 pannes
        
    def load_model(self, model_file):
        """
        Charge le modèle depuis le fichier .pkl
        Args:
            model_file: Fichier modèle (.pkl)
        Returns:
            bool: True si chargement réussi
        """
        try:
            if hasattr(model_file, 'read'):
                try:
                    model_file.seek(0)
                    self.model_package = pickle.load(model_file)
                except Exception:
                    try:
                        model_file.seek(0)
                        self.model_package = joblib.load(model_file)
                    except Exception as e:
                        raise Exception(f"Impossible de charger le modèle: {e}")
            else:
                try:
                    with open(model_file, 'rb') as f:
                        self.model_package = pickle.load(f)
                except Exception:
                    self.model_package = joblib.load(model_file)
            
            # Vérifier que c'est le bon format de modèle
            if not isinstance(self.model_package, dict):
                if hasattr(self.model_package, 'predict'):
                    temp_model = self.model_package
                    self.model_package = {
                        'regression_model': temp_model,
                        'model_name': type(temp_model).__name__,
                        'feature_columns': [],
                        'performance_metrics': {}
                    }
                else:
                    raise Exception("Le fichier ne contient pas un modèle valide")
            
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle: {str(e)}")
            return False
    
    def load_data(self, data_file):
        """
        Charge les données historiques
        Args:
            data_file: Fichier de données (.csv ou .xlsx)
        Returns:
            bool: True si chargement réussi
        """
        try:
            if hasattr(data_file, 'name'):
                if data_file.name.endswith('.csv'):
                    self.df_clean = pd.read_csv(data_file)
                elif data_file.name.endswith('.xlsx'):
                    self.df_clean = pd.read_excel(data_file)
                else:
                    st.error("Format de fichier non supporté. Utilisez .csv ou .xlsx")
                    return False
            else:
                if data_file.endswith('.csv'):
                    self.df_clean = pd.read_csv(data_file)
                elif data_file.endswith('.xlsx'):
                    self.df_clean = pd.read_excel(data_file)
            
            # ÉTAPE 1: Conversion de la colonne date
            self.df_clean['DateDebut'] = pd.to_datetime(self.df_clean['DateDebut'], errors='coerce')
            
            # ÉTAPE 2: Supprimer les lignes sans date et trier
            self.df_clean = self.df_clean.dropna(subset=['DateDebut']).copy()
            self.df_clean = self.df_clean.sort_values(['code_equipement', 'DateDebut'])
            
            # ÉTAPE 3: Identifier les pannes
            self.df_clean['is_panne'] = (self.df_clean['TypeIntervention'] == 'incident').astype(int)
            
            # ÉTAPE 4: Créer le dataset enrichi
            self._create_temporal_features()
            
            return True
        except Exception as e:
            st.error(f"Erreur lors du chargement des données: {str(e)}")
            return False
    
    def _create_temporal_features(self):
        """
        Crée le dataset avec features temporelles
        """
        enhanced_data = []
        self.valid_equipments = []
        
        for equipment in self.df_clean['code_equipement'].unique():
            eq_data = self.df_clean[self.df_clean['code_equipement'] == equipment].copy()
            
            # FILTRER UNIQUEMENT LES PANNES
            pannes = eq_data[eq_data['is_panne'] == 1].copy()
            
            # CONDITION STRICTE: Au moins 2 pannes
            if len(pannes) >= ANALYSIS_LIMITS['min_failures_for_prediction']:
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
                            (eq_data['DateDebut'] >= current_date - pd.Timedelta(days=90))
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
        
    def predict_failure(self, equipment_code, current_date=None):
        """
        Prédit la panne pour un équipement donné
        Args:
            equipment_code (str): Code de l'équipement
            current_date: Date d'analyse (par défaut: maintenant)
        Returns:
            dict: Résultat de la prédiction ou erreur
        """
        if self.model_package is None or self.df_clean is None:
            return None
        
        # Vérifier si l'équipement est dans la liste des équipements valides
        if self.valid_equipments is not None and equipment_code not in self.valid_equipments:
            return {'error': f'Équipement {equipment_code} non disponible pour prédiction (moins de 2 pannes historiques)'}
        
        if current_date is None:
            current_date = pd.Timestamp.now()
        else:
            current_date = pd.to_datetime(current_date)
        
        # Récupérer l'historique de l'équipement
        eq_history = self.df_clean[self.df_clean['code_equipement'] == equipment_code].copy()
        if len(eq_history) == 0:
            return {'error': 'Équipement non trouvé'}
        
        try:
            # Calculer toutes les features
            past_interventions = eq_history[eq_history['DateDebut'] < current_date]
            if len(past_interventions) > 0:
                days_since_last = (current_date - past_interventions['DateDebut'].max()).days
            else:
                days_since_last = 0
            
            features = {
                'age_equipment_days': max(0, (current_date - eq_history['DateDebut'].min()).days),
                'total_pannes_before': len(eq_history[(eq_history.get('is_panne', 0) == 1) & 
                                                      (eq_history['DateDebut'] < current_date)]),
                'total_interventions_before': len(eq_history[eq_history['DateDebut'] < current_date]),
                'interventions_last_7d': len(eq_history[
                    (eq_history['DateDebut'] < current_date) &
                    (eq_history['DateDebut'] >= current_date - pd.Timedelta(days=7))
                ]),
                'interventions_last_30d': len(eq_history[
                    (eq_history['DateDebut'] < current_date) &
                    (eq_history['DateDebut'] >= current_date - pd.Timedelta(days=30))
                ]),
                'interventions_last_90d': len(eq_history[
                    (eq_history['DateDebut'] < current_date) &
                    (eq_history['DateDebut'] >= current_date - pd.Timedelta(days=90))
                ]),
                'days_since_last_intervention': days_since_last,
                'month': current_date.month,
                'quarter': current_date.quarter,
                'day_of_week': current_date.dayofweek,
                'is_weekend': int(current_date.dayofweek >= 5),
                'equipement_type_encoded': 0  # Par défaut
            }
            
            # Encoder le type d'équipement si possible
            if 'label_encoder' in self.model_package and 'equipement' in eq_history.columns:
                try:
                    eq_type = eq_history['equipement'].iloc[0]
                    le = self.model_package['label_encoder']
                    if hasattr(le, 'classes_') and eq_type in le.classes_:
                        features['equipement_type_encoded'] = le.transform([eq_type])[0]
                except Exception:
                    pass
            
            # Calculer le taux d'intervention
            if features['age_equipment_days'] > 0:
                features['intervention_rate'] = features['total_interventions_before'] / features['age_equipment_days']
            else:
                features['intervention_rate'] = 0.0
            
            # Obtenir les colonnes de features du modèle
            feature_cols = self.model_package.get('feature_columns', list(features.keys()))
            
            # S'assurer que toutes les features nécessaires sont présentes
            for col in feature_cols:
                if col not in features:
                    features[col] = 0.0
            
            # Créer le DataFrame pour la prédiction
            X_pred = pd.DataFrame([features])[feature_cols]
            
            # Gérer les valeurs manquantes et infinies
            X_pred = X_pred.fillna(0)
            X_pred = X_pred.replace([np.inf, -np.inf], 0)
            
            # Prédiction avec le modèle
            model = self.model_package['regression_model']
            days_to_failure = model.predict(X_pred)[0]
            days_to_failure = max(0, days_to_failure)
            
            predicted_date = current_date + pd.Timedelta(days=int(days_to_failure))
            
            # Déterminer le niveau de risque
            risk_level, risk_class, action = self._determine_risk_level(days_to_failure)
            
            return {
                'equipment_code': equipment_code,
                'equipment_type': eq_history['equipement'].iloc[0] if 'equipement' in eq_history.columns else 'Unknown',
                'current_date': current_date.strftime('%Y-%m-%d'),
                'predicted_failure_date': predicted_date.strftime('%Y-%m-%d'),
                'days_to_failure': int(days_to_failure),
                'risk_level': risk_level,
                'risk_class': risk_class,
                'recommended_action': action,
                'failure_probabilities': DEFAULT_PROBABILITIES,
                'confidence_interval': {
                    'lower': max(0, int(days_to_failure * CONFIDENCE_INTERVAL_FACTOR['lower'])),
                    'upper': int(days_to_failure * CONFIDENCE_INTERVAL_FACTOR['upper'])
                },
                'features': features,
                'total_interventions': len(eq_history),
                'last_intervention': eq_history['DateDebut'].max().strftime('%Y-%m-%d') if len(eq_history) > 0 else 'N/A'
            }
            
        except Exception as e:
            return {'error': f'Erreur lors de la prédiction: {str(e)}'}
    
    def _determine_risk_level(self, days_to_failure):
        """
        Détermine le niveau de risque basé sur les jours avant panne
        Args:
            days_to_failure (int): Nombre de jours avant panne
        Returns:
            tuple: (risk_level, risk_class, action)
        """
        for risk_class, config in RISK_LEVELS.items():
            if days_to_failure <= config['max_days']:
                return config['name'], risk_class, config['action']
        
        # Par défaut, risque faible
        return RISK_LEVELS['low']['name'], 'low', RISK_LEVELS['low']['action']
    
    def safe_predict_batch(self, equipment_codes, max_errors=None):
        """
        Prédiction en lot avec gestion d'erreurs
        Args:
            equipment_codes (list): Liste des codes d'équipements
            max_errors (int): Nombre maximum d'erreurs autorisées
        Returns:
            list: Liste des prédictions
        """
        if max_errors is None:
            max_errors = ANALYSIS_LIMITS['max_errors']
            
        predictions = []
        error_count = 0
        
        # Filtrer les équipements valides
        if self.valid_equipments is not None:
            valid_equipment_codes = [eq for eq in equipment_codes if eq in self.valid_equipments]
            if len(valid_equipment_codes) < len(equipment_codes):
                excluded_count = len(equipment_codes) - len(valid_equipment_codes)
                st.warning(f"⚠️ {excluded_count} équipements exclus (moins de 2 pannes historiques)")
        else:
            valid_equipment_codes = equipment_codes
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, eq_code in enumerate(valid_equipment_codes):
            try:
                pred = self.predict_failure(eq_code)
                if pred and 'error' not in pred:
                    predictions.append(pred)
                else:
                    error_count += 1
                    if error_count >= max_errors:
                        st.warning(f"Trop d'erreurs de prédiction ({error_count}). Arrêt du traitement.")
                        break
            except Exception as e:
                error_count += 1
                if error_count >= max_errors:
                    st.warning(f"Trop d'erreurs de prédiction ({error_count}). Arrêt du traitement.")
                    break
            
            progress = (i + 1) / len(valid_equipment_codes)
            progress_bar.progress(progress)
            status_text.text(f"Traitement: {i + 1}/{len(valid_equipment_codes)} équipements valides")
        
        progress_bar.empty()
        status_text.empty()
        
        return predictions
