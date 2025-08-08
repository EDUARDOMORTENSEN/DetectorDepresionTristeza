import streamlit as st
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
from PIL import Image
import numpy as np
import cv2
import mediapipe as mp

# Archivo donde se guarda el registro
archivo_emociones = "registro_emociones_depresion.csv"

st.set_page_config(page_title="Detector Avanzado de Depresión", layout="centered")
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            padding: 2rem;
            border-radius: 12px;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            font-weight: bold;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 5px solid #ffc107;
        }
        .danger-box {
            background-color: #f8d7da;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 5px solid #dc3545;
        }
        .info-box {
            background-color: #d1ecf1;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 5px solid #17a2b8;
        }
        .success-box {
            background-color: #d4edda;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 5px solid #28a745;
        }
    </style>
""", unsafe_allow_html=True)

st.title(" Detector de Depresion y tristeza por rasgos faciales ")
st.write("Sistema mejorado que diferencia depresión real de expresiones similares")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5)

# ========== FUNCIONES MEJORADAS ==========
def normalizar_puntaje(valor, min_val, max_val):
    return max(0, min(100, (valor - min_val) * 100 / (max_val - min_val)))

def es_sonrisa_genuina(landmarks):
    """Detección mejorada de sonrisa Duchenne (ojos + boca)"""
    # Comisuras boca hacia arriba (puntos 291 y 61)
    comisura_boca_arriba = (landmarks[291].y < landmarks[308].y) and (landmarks[61].y < landmarks[282].y)
    
    # Arrugas en comisuras oculares (puntos 346 y 124 para "patas de gallo")
    arrugas_ojos = ((landmarks[346].y - landmarks[352].y) > 0.015) or ((landmarks[124].y - landmarks[130].y) > 0.015)
    
    # Mejillas elevadas (puntos 116 y 346)
    mejillas_elevadas = (landmarks[116].y < landmarks[118].y) and (landmarks[346].y < landmarks[348].y)
    
    return comisura_boca_arriba and arrugas_ojos and mejillas_elevadas

def detectar_musculo_corrugador(landmarks):
    """Diferencia entre depresión (cejas caídas) y enojo (cejas fruncidas)"""
    # Distancia entre ceja interna y punto medio de los ojos
    ceja_der = landmarks[65].y - landmarks[158].y
    ceja_izq = landmarks[295].y - landmarks[385].y
    return (ceja_der + ceja_izq) / 2

def analizar_ojos_depresivos(img):
    """Análisis mejorado con filtros para emociones contradictorias"""
    resultados = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not resultados.multi_face_landmarks:
        return None

    landmarks = resultados.multi_face_landmarks[0].landmark
    
    # Detección de sonrisa genuina
    es_genuina = es_sonrisa_genuina(landmarks)
    
    # Puntos clave ajustados para niños/adultos
    puntos = {
        'ceja_der': [65, 158],    # Punto interno ceja derecha
        'ceja_izq': [295, 385],    # Punto interno ceja izquierda
        'parpado_der': [159, 145], # Párpado superior derecho
        'parpado_izq': [386, 374], # Párpado superior izquierdo
        'comisura_der': [33, 133], # Comisura ocular externa derecha
        'comisura_izq': [362, 263] # Comisura ocular externa izquierda
    }

    # Cálculo de distancias normalizadas
    distancias = {}
    for nombre, [i, j] in puntos.items():
        dist = np.linalg.norm([landmarks[i].x - landmarks[j].x, landmarks[i].y - landmarks[j].y])
        distancias[nombre] = dist

    # Scores individuales con ajustes para sonrisas genuinas
    scores = {
        'cejas_caidas': (distancias['ceja_der'] + distancias['ceja_izq']) / 2,
        'parpados_pesados': (distancias['parpado_der'] + distancias['parpado_izq']) / 2,
        'ojos_caidos': (distancias['comisura_der'] + distancias['comisura_izq']) / 2,
        'corrugador': detectar_musculo_corrugador(landmarks)
    }

    # Normalización con valores empíricos mejorados
    scores_norm = {
        'cejas_caidas': normalizar_puntaje(scores['cejas_caidas'], 0.03, 0.07),
        'parpados_pesados': normalizar_puntaje(scores['parpados_pesados'], 0.02, 0.06),
        'ojos_caidos': normalizar_puntaje(scores['ojos_caidos'], 0.04, 0.08),
        'corrugador': normalizar_puntaje(scores['corrugador'], 0.01, 0.05)
    }

    # Ajuste especial para sonrisas genuinas
    if es_genuina:
        scores_norm = {k: v * 0.2 for k, v in scores_norm.items()}

    # Puntaje compuesto mejorado
    puntaje_total = (
        scores_norm['cejas_caidas'] * 0.3 +
        scores_norm['parpados_pesados'] * 0.25 +
        scores_norm['ojos_caidos'] * 0.25 +
        scores_norm['corrugador'] * 0.2
    )

    return {
        **scores_norm,
        'puntaje_total': puntaje_total,
        'cejas_estado': "Caídas" if scores_norm['cejas_caidas'] > 60 else "Normales",
        'parpados_estado': "Pesados" if scores_norm['parpados_pesados'] > 55 else "Normales",
        'comisuras_estado': "Caídas" if scores_norm['ojos_caidos'] > 65 else "Neutras",
        'sonrisa_genuina': es_genuina
    }

def filtrar_emociones(emocion, emociones, ojos_resultado):
    """Filtra falsos positivos según la emoción detectada"""
    if emocion == "happy" and ojos_resultado['sonrisa_genuina']:
        return 0.1, "✅ Sonrisa genuina detectada (Duchenne)"
    
    ajustes = {
        "angry": (0.3, "⚠️ Cejas fruncidas (enojo ≠ depresión)"),
        "surprise": (0.2, "👀 Ojos muy abiertos (sorpresa)"),
        "disgust": (0.4, "🤢 Nariz arrugada (asco)"),
        "fear": (0.5, "😨 Músculos tensos (miedo)")
    }
    
    return ajustes.get(emocion, (1.0, ""))

# ========== INTERFAZ STREAMLIT ==========
imagen_subida = st.file_uploader("Sube una foto frontal clara", type=["jpg", "jpeg", "png"])

if imagen_subida:
    try:
        imagen = Image.open(imagen_subida)
        img_array = np.array(imagen)
        st.image(imagen, caption="Imagen analizada", use_column_width=True)

        with st.spinner("Analizando con filtros mejorados..."):
            # Análisis de emociones
            try:
                resultado = DeepFace.analyze(img_path=img_array, 
                                           actions=["emotion"], 
                                           enforce_detection=False,
                                           detector_backend='mtcnn')
                emocion = resultado[0]['dominant_emotion']
                emociones = resultado[0]['emotion']
            except Exception as e:
                st.error(f"Error en análisis de emociones: {str(e)}")
                emocion = "neutral"
                emociones = {'sad': 0, 'neutral': 100}

            # Análisis ocular mejorado
            ojos_resultado = analizar_ojos_depresivos(img_array)
            
            if not ojos_resultado:
                st.error("Rostro no detectado. Usa una imagen frontal con buena iluminación.")
            else:
                # Aplicamos filtros según emoción
                factor_ajuste, mensaje = filtrar_emociones(emocion, emociones, ojos_resultado)
                if mensaje:
                    st.info(mensaje)
                
                # Cálculo final del porcentaje
                porcentaje = min(100, (
                    emociones['sad'] * 0.4 * factor_ajuste + 
                    emociones['neutral'] * 0.2 + 
                    ojos_resultado['puntaje_total'] * 0.4 * factor_ajuste
                ))

                # Evaluación mejorada
                if porcentaje > 80:
                    evaluacion = "🔴 Depresión probable"
                    recomendacion = "Consulta con un profesional de salud mental"
                    box_class = "danger-box"
                elif porcentaje > 60:
                    evaluacion = "🟠 Tristeza persistente"
                    recomendacion = "Considera buscar apoyo emocional"
                    box_class = "warning-box"
                elif porcentaje > 40:
                    evaluacion = "🟡 Estado bajo de ánimo"
                    recomendacion = "Podría ser temporal. Monitorea tus emociones"
                    box_class = "info-box"
                else:
                    evaluacion = "🟢 Estado normal"
                    recomendacion = "No se detectaron signos relevantes"
                    box_class = "success-box"

                # Resultados
                st.markdown(f"""
                <div class="{box_class}">
                    <h3>{evaluacion}</h3>
                    <p><strong>Puntaje:</strong> {porcentaje:.1f}%</p>
                    <p>{recomendacion}</p>
                    <p><strong>Detalles:</strong></p>
                    <ul>
                        <li>Emoción dominante: {emocion.upper()}</li>
                        <li>Cejas: {ojos_resultado['cejas_estado']} ({ojos_resultado['cejas_caidas']:.1f}%)</li>
                        <li>Párpados: {ojos_resultado['parpados_estado']} ({ojos_resultado['parpados_pesados']:.1f}%)</li>
                        <li>Comisuras: {ojos_resultado['comisuras_estado']} ({ojos_resultado['ojos_caidos']:.1f}%)</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {str(e)}")