"""
Einfache Gradio-App für Plant Disease Detection
"""

import gradio as gr
import numpy as np
from PIL import Image


def predict_disease(image):
    """Temporäre Funktion für die Krankheitserkennung"""
    if image is None:
        return "Bitte laden Sie ein Bild hoch."

    # Demo-Vorhersage (später durch echtes ML-Model ersetzen)
    predictions = [
        "Gesunde Pflanze 🌱",
        "Blattfleckenkrankheit 🍃",
        "Mehltau 🍄",
        "Rostpilz 🦠",
        "Virusbefall 🔴",
    ]

    # Zufällige Demo-Vorhersage
    import random

    prediction = random.choice(predictions)
    confidence = random.uniform(0.75, 0.99)

    return f"Vorhersage: {prediction}\nGenauigkeit: {confidence:.2%}"


def main():
    """Hauptfunktion für die Gradio-App"""

    # Erstelle Gradio Interface
    interface = gr.Interface(
        fn=predict_disease,
        inputs=gr.Image(type="pil", label="Pflanzenbild hochladen"),
        outputs=gr.Textbox(label="Diagnose", lines=3),
        title="🌱 Plant Disease Detection",
        description="Laden Sie ein Bild einer Pflanze hoch, um Krankheiten zu erkennen.",
        theme=gr.themes.Soft(),
        examples=None,
    )

    # Starte die App
    interface.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
