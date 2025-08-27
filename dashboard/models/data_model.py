#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modèles de données pour l'application
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class Equipment:
    """Modèle pour un équipement"""
    code: str
    type: str
    total_interventions: int
    total_failures: int
    last_intervention_date: Optional[datetime] = None
    age_days: int = 0

@dataclass
class PredictionResult:
    """Modèle pour un résultat de prédiction"""
    equipment_code: str
    equipment_type: str
    current_date: str
    predicted_failure_date: str
    days_to_failure: int
    risk_level: str
    risk_class: str
    recommended_action: str
    failure_probabilities: Dict[str, float]
    confidence_interval: Dict[str, int]
    features: Dict[str, Any]
    total_interventions: int
    last_intervention: str

@dataclass
class ModelInfo:
    """Modèle pour les informations du modèle"""
    name: str
    feature_count: int
    training_date: Optional[str] = None
    training_samples: Optional[int] = None
    performance_metrics: Optional[Dict[str, float]] = None

@dataclass
class User:
    """Modèle pour un utilisateur"""
    username: str
    name: str
    role: str
    authenticated: bool = False

@dataclass
class AnalysisConfig:
    """Modèle pour la configuration d'analyse"""
    max_equipments: int
    analysis_date: datetime
    include_critical_only: bool = False
    sort_by_priority: bool = True

@dataclass
class DatasetInfo:
    """Modèle pour les informations du dataset"""
    total_records: int
    unique_equipments: int
    valid_equipments: int
    period_days: int
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    failure_rate: float = 0.0
