"""
GeoMech Pro - Frontend Configuration
=======================================
Branding / contact info shown in the Streamlit UI.

Change these values here (or override them via environment variables)
instead of editing app.py directly -- this is the single place that
controls the developer/contact info shown across the dashboard.
"""
import os

PLATFORM_NAME = os.environ.get("GEOMECH_PLATFORM_NAME", "GeoMech Pro SaaS Platform")
DEVELOPER_NAME = os.environ.get("GEOMECH_DEVELOPER_NAME", "Mehdi Ghaidri")
DEVELOPER_TITLE = os.environ.get("GEOMECH_DEVELOPER_TITLE", "Petroleum Engineer")
DEVELOPER_EMAIL = os.environ.get("GEOMECH_DEVELOPER_EMAIL", "ghadirimehdi84@gmail.com")
DEVELOPER_WHATSAPP = os.environ.get("GEOMECH_DEVELOPER_WHATSAPP", "+98-913-017-7748")
