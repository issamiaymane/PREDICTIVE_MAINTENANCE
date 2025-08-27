#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Styles CSS pour l'interface utilisateur
"""

def get_css_styles():
    """
    Retourne les styles CSS pour l'application
    Returns:
        str: CSS styles
    """
    return """
    <style>
        .main-header {
            background: linear-gradient(135deg, #1e4a72 0%, #c41e3a 50%, #e67e22 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.9;
            margin: 0;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-left: 5px solid #1e4a72;
            margin-bottom: 1rem;
            transition: transform 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .risk-critical {
            background: linear-gradient(135deg, #c41e3a 0%, #e74c3c 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
        }
        
        .risk-high {
            background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(230, 126, 34, 0.3);
        }
        
        .risk-moderate {
            background: linear-gradient(135deg, #f39c12 0%, #f1c40f 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(241, 196, 15, 0.3);
        }
        
        .risk-low {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 4px 12px rgba(39, 174, 96, 0.3);
        }
        
        .prediction-result {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 2px solid #1e4a72;
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            box-shadow: 0 8px 24px rgba(0,0,0,0.1);
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #1e4a72 0%, #c41e3a 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(30, 74, 114, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(30, 74, 114, 0.4);
        }
        
        .info-card {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            border-top: 4px solid #e67e22;
            margin-bottom: 1rem;
        }
        
        .success-card {
            background: linear-gradient(135deg, #d5f4e6 0%, #f0fff4 100%);
            border: 2px solid #27ae60;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            color: #1a5832;
            font-weight: 600;
        }
        
        .warning-card {
            background: linear-gradient(135deg, #fef5e7 0%, #fff8f0 100%);
            border: 2px solid #e67e22;
            border-radius: 12px;
            padding: 1rem;
            margin: 1rem 0;
            color: #8b4513;
            font-weight: 600;
        }
        
        .color-dot {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .color-blue { background-color: #1e4a72; }
        .color-red { background-color: #c41e3a; }
        .color-orange { background-color: #e67e22; }
        .color-green { background-color: #27ae60; }
        
        .snrt-colors {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 1rem;
        }

        .login {
            text-align: center;
        }
        
        .page-header {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            border-left: 5px solid #1e4a72;
        }
        
        .page-header h2 {
            color: #1e4a72;
            margin-bottom: 0.5rem;
            font-size: 2rem;
            font-weight: 600;
        }
        
        .page-header p {
            color: #7f8c8d;
            margin: 0;
            font-size: 1.1rem;
        }
        
        /* Style pour le bouton de déconnexion */
        .logout-button {
            background: linear-gradient(135deg, #c41e3a 0%, #e74c3c 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .logout-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
        }
    </style>
    """

def apply_styles():
    """
    Applique les styles CSS à l'application Streamlit
    """
    import streamlit as st
    st.markdown(get_css_styles(), unsafe_allow_html=True)
