#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Traitement des données
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame en appliquant les transformations de base
    Args:
        df (pd.DataFrame): DataFrame à nettoyer
    Returns:
        pd.DataFrame: DataFrame nettoyé
    """
    # Copie pour éviter la modification de l'original
    df_clean = df.copy()
    
    # Conversion de la colonne date
    if 'DateDebut' in df_clean.columns:
        df_clean['DateDebut'] = pd.to_datetime(df_clean['DateDebut'], errors='coerce')
    
    # Supprimer les lignes sans date et trier
    df_clean = df_clean.dropna(subset=['DateDebut'])
    df_clean = df_clean.sort_values(['code_equipement', 'DateDebut'])
    
    # Identifier les pannes
    if 'TypeIntervention' in df_clean.columns:
        df_clean['is_panne'] = (df_clean['TypeIntervention'] == 'incident').astype(int)
    
    return df_clean

def calculate_temporal_features(df: pd.DataFrame, equipment_code: str, current_date: datetime) -> Dict:
    """
    Calcule les features temporelles pour un équipement donné
    Args:
        df (pd.DataFrame): DataFrame des données
        equipment_code (str): Code de l'équipement
        current_date (datetime): Date d'analyse
    Returns:
        Dict: Features calculées
    """
    eq_history = df[df['code_equipement'] == equipment_code].copy()
    
    if len(eq_history) == 0:
        return {}
    
    # Calculer les features de base
    past_interventions = eq_history[eq_history['DateDebut'] < current_date]
    days_since_last = 0
    
    if len(past_interventions) > 0:
        days_since_last = (current_date - past_interventions['DateDebut'].max()).days
    
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
        'is_weekend': int(current_date.dayofweek >= 5)
    }
    
    # Calculer le taux d'intervention
    if features['age_equipment_days'] > 0:
        features['intervention_rate'] = features['total_interventions_before'] / features['age_equipment_days']
    else:
        features['intervention_rate'] = 0.0
    
    return features

def validate_equipment_for_prediction(df: pd.DataFrame, equipment_code: str, min_failures: int = 2) -> bool:
    """
    Valide si un équipement peut être utilisé pour la prédiction
    Args:
        df (pd.DataFrame): DataFrame des données
        equipment_code (str): Code de l'équipement
        min_failures (int): Nombre minimum de pannes requis
    Returns:
        bool: True si l'équipement est valide
    """
    eq_data = df[df['code_equipement'] == equipment_code]
    if len(eq_data) == 0:
        return False
    
    # Compter les pannes
    if 'is_panne' in eq_data.columns:
        failure_count = eq_data['is_panne'].sum()
    else:
        # Fallback: compter les incidents
        failure_count = len(eq_data[eq_data['TypeIntervention'] == 'incident'])
    
    return failure_count >= min_failures

def get_valid_equipments(df: pd.DataFrame, min_failures: int = 2) -> List[str]:
    """
    Obtient la liste des équipements valides pour la prédiction
    Args:
        df (pd.DataFrame): DataFrame des données
        min_failures (int): Nombre minimum de pannes requis
    Returns:
        List[str]: Liste des codes d'équipements valides
    """
    valid_equipments = []
    
    for equipment in df['code_equipement'].unique():
        if validate_equipment_for_prediction(df, equipment, min_failures):
            valid_equipments.append(equipment)
    
    return valid_equipments

def create_enhanced_dataset(df: pd.DataFrame, min_failures: int = 2) -> Tuple[pd.DataFrame, List[str]]:
    """
    Crée le dataset enrichi avec features temporelles
    Args:
        df (pd.DataFrame): DataFrame des données
        min_failures (int): Nombre minimum de pannes requis
    Returns:
        Tuple[pd.DataFrame, List[str]]: Dataset enrichi et liste des équipements valides
    """
    enhanced_data = []
    valid_equipments = []
    
    for equipment in df['code_equipement'].unique():
        eq_data = df[df['code_equipement'] == equipment].copy()
        
        # Filtrer uniquement les pannes
        pannes = eq_data[eq_data['is_panne'] == 1].copy()
        
        # Vérifier le critère de validité
        if len(pannes) >= min_failures:
            valid_equipments.append(equipment)
            
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
    enhanced_df = pd.DataFrame(enhanced_data)
    
    return enhanced_df, valid_equipments
