"""
GeoMech Pro - Backend Configuration
=====================================
Branding / contact info used in generated PDF reports.

Change these values here (or override them via environment variables)
instead of editing report_generator.py directly -- this is the single
place that controls what shows up on every report.
"""
import os

PLATFORM_NAME = os.environ.get("GEOMECH_PLATFORM_NAME", "GeoMech Pro SaaS Platform")
DEVELOPER_NAME = os.environ.get("GEOMECH_DEVELOPER_NAME", "Mehdi Ghaidri")
DEVELOPER_EMAIL = os.environ.get("GEOMECH_DEVELOPER_EMAIL", "ghadirimehdi84@gmail.com")
DEVELOPER_WHATSAPP = os.environ.get("GEOMECH_DEVELOPER_WHATSAPP", "+98-913-017-7748")
