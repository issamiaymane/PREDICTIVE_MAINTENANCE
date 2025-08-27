#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fonctions d'export
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

def export_predictions_to_csv(predictions: List[Dict], filename: Optional[str] = None) -> str:
    """
    Exporte les prédictions vers un fichier CSV
    Args:
        predictions (List[Dict]): Liste des prédictions
        filename (Optional[str]): Nom du fichier (optionnel)
    Returns:
        str: Contenu CSV
    """
    if not predictions:
        return ""
    
    df = pd.DataFrame(predictions)
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"predictions_dashboard_{timestamp}.csv"
    
    return df.to_csv(index=False)

def export_critical_equipments_to_csv(predictions: List[Dict], filename: Optional[str] = None) -> str:
    """
    Exporte uniquement les équipements critiques vers un fichier CSV
    Args:
        predictions (List[Dict]): Liste des prédictions
        filename (Optional[str]): Nom du fichier (optionnel)
    Returns:
        str: Contenu CSV
    """
    if not predictions:
        return ""
    
    df = pd.DataFrame(predictions)
    critical_df = df[df['risk_level'] == 'CRITIQUE']
    
    if len(critical_df) == 0:
        return ""
    
    if filename is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"critiques_snrt_{timestamp}.csv"
    
    return critical_df.to_csv(index=False)

def create_summary_report(predictions: List[Dict]) -> Dict:
    """
    Crée un rapport de synthèse des prédictions
    Args:
        predictions (List[Dict]): Liste des prédictions
    Returns:
        Dict: Rapport de synthèse
    """
    if not predictions:
        return {}
    
    df = pd.DataFrame(predictions)
    
    # Statistiques de base
    total_equipments = len(df)
    critical_count = len(df[df['days_to_failure'] <= 7])
    high_count = len(df[(df['days_to_failure'] > 7) & (df['days_to_failure'] <= 30)])
    moderate_count = len(df[(df['days_to_failure'] > 30) & (df['days_to_failure'] <= 90)])
    low_count = len(df[df['days_to_failure'] > 90])
    
    # Statistiques temporelles
    avg_days_to_failure = df['days_to_failure'].mean()
    median_days_to_failure = df['days_to_failure'].median()
    min_days_to_failure = df['days_to_failure'].min()
    max_days_to_failure = df['days_to_failure'].max()
    
    # Équipements les plus critiques
    top_critical = df.nsmallest(10, 'days_to_failure')[['equipment_code', 'equipment_type', 'days_to_failure', 'predicted_failure_date']]
    
    report = {
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_equipments_analyzed': total_equipments,
        'risk_distribution': {
            'critical': {'count': critical_count, 'percentage': (critical_count/total_equipments)*100},
            'high': {'count': high_count, 'percentage': (high_count/total_equipments)*100},
            'moderate': {'count': moderate_count, 'percentage': (moderate_count/total_equipments)*100},
            'low': {'count': low_count, 'percentage': (low_count/total_equipments)*100}
        },
        'temporal_statistics': {
            'average_days_to_failure': avg_days_to_failure,
            'median_days_to_failure': median_days_to_failure,
            'min_days_to_failure': min_days_to_failure,
            'max_days_to_failure': max_days_to_failure
        },
        'top_critical_equipments': top_critical.to_dict('records')
    }
    
    return report

def export_summary_report_to_text(report: Dict) -> str:
    """
    Exporte le rapport de synthèse en format texte
    Args:
        report (Dict): Rapport de synthèse
    Returns:
        str: Rapport au format texte
    """
    if not report:
        return "Aucun rapport disponible"
    
    text = f"""
RAPPORT DE SYNTHÈSE - SYSTÈME DE MAINTENANCE PRÉDICTIVE INTELLIGENTE
====================================================================

Date de génération: {report['generation_date']}
Total d'équipements analysés: {report['total_equipments_analyzed']:,}

RÉPARTITION DES RISQUES:
-----------------------
• Critique (≤ 7 jours): {report['risk_distribution']['critical']['count']:,} équipements ({report['risk_distribution']['critical']['percentage']:.1f}%)
• Élevé (8-30 jours): {report['risk_distribution']['high']['count']:,} équipements ({report['risk_distribution']['high']['percentage']:.1f}%)
• Modéré (31-90 jours): {report['risk_distribution']['moderate']['count']:,} équipements ({report['risk_distribution']['moderate']['percentage']:.1f}%)
• Faible (> 90 jours): {report['risk_distribution']['low']['count']:,} équipements ({report['risk_distribution']['low']['percentage']:.1f}%)

STATISTIQUES TEMPORELLES:
------------------------
• Délai moyen avant panne: {report['temporal_statistics']['average_days_to_failure']:.1f} jours
• Délai médian avant panne: {report['temporal_statistics']['median_days_to_failure']:.1f} jours
• Délai minimum: {report['temporal_statistics']['min_days_to_failure']:.0f} jours
• Délai maximum: {report['temporal_statistics']['max_days_to_failure']:.0f} jours

TOP 10 ÉQUIPEMENTS CRITIQUES:
----------------------------
"""
    
    for i, eq in enumerate(report['top_critical_equipments'], 1):
        text += f"{i:2d}. {eq['equipment_code']} ({eq['equipment_type']}) - {eq['days_to_failure']} jours - {eq['predicted_failure_date']}\n"
    
    text += "\n" + "="*70 + "\n"
    text += "SNRT - DISI - Système de Maintenance Prédictive Intelligente\n"
    
    return text

def export_to_excel(predictions: List[Dict], filename: Optional[str] = None) -> bytes:
    """
    Exporte les prédictions vers un fichier Excel
    Args:
        predictions (List[Dict]): Liste des prédictions
        filename (Optional[str]): Nom du fichier (optionnel)
    Returns:
        bytes: Contenu du fichier Excel
    """
    if not predictions:
        return b""
    
    df = pd.DataFrame(predictions)
    
    # Créer un writer Excel
    output = pd.ExcelWriter('temp.xlsx', engine='openpyxl')
    
    # Feuille principale avec toutes les prédictions
    df.to_excel(output, sheet_name='Toutes les Prédictions', index=False)
    
    # Feuille avec les équipements critiques seulement
    critical_df = df[df['risk_level'] == 'CRITIQUE']
    if len(critical_df) > 0:
        critical_df.to_excel(output, sheet_name='Équipements Critiques', index=False)
    
    # Feuille avec le rapport de synthèse
    report = create_summary_report(predictions)
    if report:
        report_data = []
        report_data.append(['RAPPORT DE SYNTHÈSE'])
        report_data.append(['Date de génération', report['generation_date']])
        report_data.append(['Total équipements analysés', report['total_equipments_analyzed']])
        report_data.append([])
        report_data.append(['RÉPARTITION DES RISQUES'])
        for risk, data in report['risk_distribution'].items():
            report_data.append([f"{risk.title()}", f"{data['count']} ({data['percentage']:.1f}%)"])
        report_data.append([])
        report_data.append(['STATISTIQUES TEMPORELLES'])
        for stat, value in report['temporal_statistics'].items():
            report_data.append([stat.replace('_', ' ').title(), f"{value:.1f}"])
        
        report_df = pd.DataFrame(report_data)
        report_df.to_excel(output, sheet_name='Rapport de Synthèse', index=False, header=False)
    
    output.close()
    
    # Lire le fichier et le retourner
    with open('temp.xlsx', 'rb') as f:
        excel_content = f.read()
    
    # Nettoyer le fichier temporaire
    import os
    os.remove('temp.xlsx')
    
    return excel_content
