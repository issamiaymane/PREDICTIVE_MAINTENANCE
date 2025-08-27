# Système de Maintenance Prédictive Intelligente - SNRT

## Vue d'ensemble

Ce projet implémente un système complet de maintenance prédictive pour la SNRT (Société Nationale de Radiodiffusion et de Télévision). Le système utilise l'intelligence artificielle pour prédire les défaillances d'équipements et optimiser les plans de maintenance.

## Architecture du Projet

```
PREDICTIVE_MAINTENANCE/
├── dashboard/          # Interface web Streamlit
│   ├── app.py          # Application principale
│   ├── auth/           # Authentification et autorisation
│   ├── config/         # Configuration système
│   ├── models/         # Modèles de données et prédiction
│   ├── modules/        # Modules fonctionnels
│   ├── ui/             # Composants d'interface
│   ├── utils/          # Utilitaires
│   └── requirements.txt
├── alerts/             # Système d'alertes automatiques
│   ├── main.py         # Générateur d'alertes
│   ├── config.json     # Configuration des alertes
│   └── requirements.txt
├── notebooks/          # Développement et analyse
│   ├── dataset.ipynb   # Analyse des données
│   ├── model.ipynb     # Développement du modèle
│   ├── data/           # Données sources
│   └── requirements.txt
└── README.md          # Documentation principale
```

## Composants Principaux

### 1. **Dashboard Web** (`/dashboard`)
Interface utilisateur moderne pour :
- **Prédictions en temps réel** : Analyse des équipements
- **Gestion des alertes** : Suivi des niveaux de risque
- **Analyse des données** : Visualisations et rapports
- **Administration** : Gestion des utilisateurs et configurations

### 2. **Système d'Alertes** (`/alerts`)
Automatisation des notifications :
- **Analyse automatique** des prédictions
- **Génération d'alertes** selon les niveaux de risque
- **Logging et historique** des actions
- **Export des résultats** en format standardisé

### 3. **Notebooks de Développement** (`/notebooks`)
Environnement de développement et recherche :
- **Exploration des données** : Analyse statistique
- **Développement de modèles** : Entraînement et validation
- **Tests et expérimentations** : Validation des approches

## Fonctionnalités Clés

###  **Prédiction Intelligente**
- Modèles de machine learning avancés
- Prédiction du temps avant défaillance
- Niveaux de confiance pour chaque prédiction
- Mise à jour automatique des modèles

###  **Système d'Alertes**
- **Critique** : Maintenance d'urgence (< 7 jours)
- **Élevé** : Planification urgente (7-30 jours)
- **Modéré** : Surveillance renforcée (30-90 jours)
- **Faible** : Maintenance préventive (> 90 jours)

###  **Analytics Avancés**
- Visualisations interactives
- Rapports automatisés
- Analyse des tendances
- Métriques de performance

###  **Sécurité et Gestion des Accès**
- Authentification sécurisée
- Gestion des rôles et permissions
- Traçabilité des actions
- Protection des données sensibles

## Technologies Utilisées

### Backend
- **Python 3.8+** : Langage principal
- **Streamlit** : Interface web
- **Pandas & NumPy** : Traitement des données
- **Scikit-learn** : Machine learning
- **MongoDB** : Base de données

### Frontend
- **Streamlit Components** : Interface utilisateur
- **Plotly** : Visualisations interactives
- **CSS personnalisé** : Design moderne

### DevOps
- **Git** : Versioning
- **Cron** : Automatisation des tâches

## Installation et Démarrage

### Prérequis
- Python 3.8 ou supérieur
- MongoDB

### Installation Rapide

```bash
# 1. Cloner le repository
git clone <repository-url>
cd PREDICTIVE_MAINTENANCE

# 2. Installer les dépendances du dashboard
cd dashboard
pip install -r requirements.txt

# 3. Installer les dépendances des alertes
cd ../alerts
pip install -r requirements.txt

# 4. Installer les dépendances des notebooks
cd ../notebooks
pip install -r requirements.txt
```

### Démarrage du Dashboard

```bash
cd dashboard
streamlit run app.py
```

### Configuration des Alertes

```bash
cd alerts
python main.py
```

## Configuration

### Variables d'Environnement

```bash
# Configuration MongoDB
MONGODB_URI=mongodb://localhost:27017
MONGODB_DB=maintenance_predictive

# Configuration des alertes
ALERTS_CONFIG_PATH=./alerts/config.json
LOG_LEVEL=INFO
```

### Fichiers de Configuration

- **Dashboard** : `dashboard/config/settings.py`
- **Base de données** : `dashboard/config/database.py`
- **Alertes** : `alerts/config.json`

## Utilisation

### Pour les Utilisateurs Finaux

1. **Accès au Dashboard**
   - Ouvrir l'URL du dashboard
   - Se connecter avec ses identifiants
   - Consulter les prédictions et alertes

2. **Consultation des Alertes**
   - Voir les équipements à risque
   - Consulter les recommandations
   - Exporter les rapports

### Pour les Administrateurs

1. **Gestion des Utilisateurs**
   - Ajout/suppression d'utilisateurs
   - Attribution des rôles
   - Surveillance des accès

2. **Configuration du Système**
   - Ajustement des seuils d'alerte
   - Configuration des modèles
   - Gestion des données

### Pour les Développeurs

1. **Développement de Modèles**
   - Utiliser les notebooks Jupyter
   - Tester de nouveaux algorithmes
   - Valider les performances

2. **Intégration de Nouvelles Fonctionnalités**
   - Suivre l'architecture modulaire
   - Respecter les bonnes pratiques
   - Documenter les changements

## Maintenance et Support

### Surveillance Continue
- Monitoring des performances du modèle
- Surveillance des alertes générées
- Vérification de la qualité des données

### Mises à Jour
- Réentraînement périodique des modèles
- Mise à jour des dépendances
- Amélioration des fonctionnalités

### Support Technique
- Documentation complète dans chaque module
- Logs détaillés pour le débogage
- Procédures de sauvegarde et restauration

## Sécurité

### Protection des Données
- Chiffrement des données sensibles
- Authentification sécurisée
- Gestion des sessions

### Contrôle d'Accès
- Rôles et permissions granulaires
- Audit des actions utilisateurs
- Isolation des environnements

## Performance

### Optimisations
- Cache des prédictions fréquentes
- Traitement asynchrone des alertes
- Optimisation des requêtes de base de données

### Monitoring
- Métriques de performance en temps réel
- Alertes sur les dégradations
- Rapports de performance automatisés

## Contribution

### Guidelines
- Suivre les conventions de code Python
- Documenter les nouvelles fonctionnalités
- Tester avant soumission

### Processus
1. Fork du repository
2. Création d'une branche feature
3. Développement et tests
4. Pull request avec documentation

## Licence

Ce projet est développé pour la SNRT - DISI.
Tous droits réservés.

---

**Auteur** : Aymane ISSAMI  
**Organisation** : SNRT - DISI  
**Version** : 1.0  
**Date** : 08/2025  
**Contact** : [issami.aymane@gmail.com]
