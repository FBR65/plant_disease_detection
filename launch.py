"""
Plant Disease Detection - Unified Launch Script
Startet alle Systemkomponenten koordiniert
"""

import asyncio
import logging
import argparse
import subprocess
import sys
import time
import os
from pathlib import Path
import psutil

# .env-Datei laden
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Fallback ohne python-dotenv
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

# Logging-Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent


class SystemLauncher:
    """Koordinierter Start aller System-Komponenten"""

    def __init__(self):
        self.qdrant_process = None
        self.services_running = []

    def check_qdrant_available(self) -> bool:
        """Prüft ob Qdrant erreichbar ist"""
        try:
            import requests

            response = requests.get("http://localhost:6333/collections", timeout=5)
            return response.status_code == 200
        except:
            return False

    def start_qdrant_docker(self) -> bool:
        """Startet Qdrant via Docker"""
        logger.info("🐳 Starte Qdrant via Docker...")

        try:
            # Prüfe ob Docker verfügbar ist
            subprocess.run(["docker", "--version"], check=True, capture_output=True)

            # Prüfe ob Qdrant bereits läuft
            if self.check_qdrant_available():
                logger.info("✅ Qdrant bereits verfügbar")
                return True

            # Starte Qdrant Container
            cmd = [
                "docker",
                "run",
                "-d",
                "--name",
                "qdrant-plant-disease",
                "-p",
                "6333:6333",
                "-p",
                "6334:6334",
                "-v",
                "qdrant_storage:/qdrant/storage",
                "qdrant/qdrant",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                logger.info("🚀 Qdrant-Container gestartet")

                # Warte auf Verfügbarkeit
                for i in range(30):  # 30 Sekunden warten
                    if self.check_qdrant_available():
                        logger.info("✅ Qdrant ist verfügbar")
                        return True
                    time.sleep(1)

                logger.error(
                    "❌ Qdrant-Timeout - Container läuft aber nicht erreichbar"
                )
                return False
            else:
                # Container existiert möglicherweise bereits
                if "already in use" in result.stderr:
                    logger.info("📦 Qdrant-Container bereits vorhanden - starte neu...")
                    subprocess.run(["docker", "start", "qdrant-plant-disease"])
                    time.sleep(5)
                    return self.check_qdrant_available()
                else:
                    logger.error(f"❌ Docker-Fehler: {result.stderr}")
                    return False

        except FileNotFoundError:
            logger.error("❌ Docker nicht installiert oder nicht verfügbar")
            logger.info("💡 Installiere Docker oder starte Qdrant manuell")
            return False
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler beim Qdrant-Start: {e}")
            return False

    async def setup_qdrant_data(self, force_recreate: bool = False) -> bool:
        """Lädt Daten in Qdrant wenn nötig"""
        logger.info("📊 Prüfe Qdrant-Datenbestand...")

        try:
            # Importiere Setup-Script
            sys.path.append(str(PROJECT_ROOT / "scripts"))
            from setup_qdrant import setup_qdrant_database

            dataset_path = PROJECT_ROOT / "data" / "PlantDoc"
            if not dataset_path.exists():
                logger.warning(f"⚠️ Dataset nicht gefunden: {dataset_path}")
                logger.info(
                    "💡 Lade PlantDoc-Dataset herunter und platziere es in data/PlantDoc/"
                )
                return False

            await setup_qdrant_database(str(dataset_path), force_recreate)
            logger.info("✅ Qdrant-Daten verfügbar")
            return True

        except Exception as e:
            logger.error(f"❌ Fehler beim Qdrant-Setup: {e}")
            return False

    def check_environment_variables(self) -> bool:
        """Prüft wichtige Umgebungsvariablen für LLM"""
        logger.info("🔧 Prüfe Umgebungsvariablen...")

        # LLM-Konfiguration (optional)
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("API_KEY")
        model_name = os.getenv("MODEL_NAME")

        if not base_url:
            logger.warning("⚠️ BASE_URL nicht gesetzt - LLM im Demo-Modus")
            logger.info("💡 Setze BASE_URL, API_KEY, MODEL_NAME für LLM-Integration")
            return False
        else:
            logger.info(f"✅ LLM konfiguriert: {model_name} @ {base_url}")
            return True

    def start_gradio_app(self):
        """Startet die Gradio-Anwendung"""
        logger.info("🌐 Starte Gradio-App...")

        try:
            # Projekt-Root als Working Directory
            os.chdir(PROJECT_ROOT)

            # Gradio-App starten
            sys.path.append(str(PROJECT_ROOT / "src"))
            from gradio_app import main

            main()

        except KeyboardInterrupt:
            logger.info("⏹️ Gradio-App gestoppt")
        except Exception as e:
            logger.error(f"❌ Fehler beim Starten der Gradio-App: {e}")
            raise

    def cleanup(self):
        """Aufräumen beim Beenden"""
        logger.info("🧹 Aufräumen...")

        # Qdrant-Container stoppen (optional)
        # if self.qdrant_process:
        #     logger.info("🛑 Stoppe Qdrant-Container...")
        #     subprocess.run(["docker", "stop", "qdrant-plant-disease"])

    async def launch_full_system(self, args):
        """Startet das komplette System"""
        logger.info("🚀 PLANT DISEASE DETECTION - SYSTEM START")
        logger.info("=" * 60)

        success_steps = 0
        total_steps = 4

        try:
            # Schritt 1: Qdrant starten
            if args.start_qdrant or not self.check_qdrant_available():
                if self.start_qdrant_docker():
                    success_steps += 1
                else:
                    logger.error("❌ Qdrant-Start fehlgeschlagen")
            else:
                logger.info("✅ Qdrant bereits verfügbar")
                success_steps += 1

            # Schritt 2: Umgebung prüfen
            if self.check_environment_variables():
                success_steps += 1
            else:
                success_steps += 0.5  # Teilweise OK

            # Schritt 3: Qdrant-Daten laden
            if args.setup_data or args.force_data_reload:
                if await self.setup_qdrant_data(args.force_data_reload):
                    success_steps += 1
                else:
                    logger.warning("⚠️ Qdrant-Daten-Setup unvollständig")
            else:
                logger.info("⏭️ Qdrant-Daten-Setup übersprungen")
                success_steps += 1

            # Schritt 4: Gradio-App starten
            if success_steps >= 2.5:  # Mindestanforderungen erfüllt
                logger.info("🎯 System bereit - starte Anwendung...")
                logger.info("=" * 60)
                success_steps += 1

                self.start_gradio_app()
            else:
                logger.error("❌ Zu viele Systemfehler - Anwendung nicht gestartet")

        except KeyboardInterrupt:
            logger.info("⏹️ System-Start abgebrochen")
        except Exception as e:
            logger.error(f"❌ Unerwarteter Fehler: {e}")
        finally:
            self.cleanup()
            logger.info(f"📊 Erfolgreich: {success_steps}/{total_steps} Schritte")


async def main():
    parser = argparse.ArgumentParser(
        description="Plant Disease Detection - Unified System Launcher"
    )

    parser.add_argument(
        "--start-qdrant", action="store_true", help="Starte Qdrant-Container"
    )

    parser.add_argument(
        "--setup-data", action="store_true", help="Lade Dataset in Qdrant"
    )

    parser.add_argument(
        "--force-data-reload",
        action="store_true",
        help="Erzwinge Neuladen der Daten (überschreibt existierende)",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Nur System-Check, keine Anwendung starten",
    )

    args = parser.parse_args()

    launcher = SystemLauncher()

    if args.check_only:
        logger.info("🔍 System-Check...")
        qdrant_ok = launcher.check_qdrant_available()
        env_ok = launcher.check_environment_variables()

        logger.info("📊 SYSTEM-STATUS:")
        logger.info(f"  🗄️ Qdrant: {'✅ OK' if qdrant_ok else '❌ Nicht verfügbar'}")
        logger.info(f"  🧠 LLM: {'✅ Konfiguriert' if env_ok else '⚠️ Demo-Modus'}")

        return

    await launcher.launch_full_system(args)


if __name__ == "__main__":
    asyncio.run(main())
