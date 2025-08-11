import streamlit as st
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
import numpy as np
import cv2
import mediapipe as mp
import time

# Archivo donde se guarda el registro
archivo_emociones = "registro_emociones_automatico.csv"

st.set_page_config(page_title="Detector Automático de Tristeza", layout="centered")
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
        .video-container {
            display: flex;
            justify-content: center;
            margin: 20px 0;
        }
        .results-container {
            margin-top: 20px;
        }
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px;
            background-color: #ff4444;
            color: white;
            border-radius: 5px;
            z-index: 1000;
            animation: fadeIn 0.5s;
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

st.title("Detector Automático de Tristeza en Tiempo Real")
st.write("Sistema que monitorea continuamente expresiones faciales y registra automáticamente indicios de tristeza")

# Configuración de MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, 
                                 max_num_faces=1,
                                 refine_landmarks=True,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

# ========== FUNCIONES DE ANÁLISIS ==========
def normalizar_puntaje(valor, min_val, max_val):
    return max(0, min(100, (valor - min_val) * 100 / (max_val - min_val)))

def es_sonrisa_genuina(landmarks):
    comisura_boca_arriba = (landmarks[291].y < landmarks[308].y) and (landmarks[61].y < landmarks[282].y)
    arrugas_ojos = ((landmarks[346].y - landmarks[352].y) > 0.015) or ((landmarks[124].y - landmarks[130].y) > 0.015)
    mejillas_elevadas = (landmarks[116].y < landmarks[118].y) and (landmarks[346].y < landmarks[348].y)
    return comisura_boca_arriba and arrugas_ojos and mejillas_elevadas

def detectar_musculo_corrugador(landmarks):
    ceja_der = landmarks[65].y - landmarks[158].y
    ceja_izq = landmarks[295].y - landmarks[385].y
    return (ceja_der + ceja_izq) / 2

def analizar_ojos_depresivos(img):
    resultados = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not resultados.multi_face_landmarks:
        return None

    landmarks = resultados.multi_face_landmarks[0].landmark
    es_genuina = es_sonrisa_genuina(landmarks)
    
    puntos = {
        'ceja_der': [65, 158],
        'ceja_izq': [295, 385],
        'parpado_der': [159, 145],
        'parpado_izq': [386, 374],
        'comisura_der': [33, 133],
        'comisura_izq': [362, 263]
    }

    distancias = {}
    for nombre, [i, j] in puntos.items():
        dist = np.linalg.norm([landmarks[i].x - landmarks[j].x, landmarks[i].y - landmarks[j].y])
        distancias[nombre] = dist

    scores = {
        'cejas_caidas': (distancias['ceja_der'] + distancias['ceja_izq']) / 2,
        'parpados_pesados': (distancias['parpado_der'] + distancias['parpado_izq']) / 2,
        'ojos_caidos': (distancias['comisura_der'] + distancias['comisura_izq']) / 2,
        'corrugador': detectar_musculo_corrugador(landmarks)
    }

    scores_norm = {
        'cejas_caidas': normalizar_puntaje(scores['cejas_caidas'], 0.03, 0.07),
        'parpados_pesados': normalizar_puntaje(scores['parpados_pesados'], 0.02, 0.06),
        'ojos_caidos': normalizar_puntaje(scores['ojos_caidos'], 0.04, 0.08),
        'corrugador': normalizar_puntaje(scores['corrugador'], 0.01, 0.05)
    }

    if es_genuina:
        scores_norm = {k: v * 0.2 for k, v in scores_norm.items()}

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
    if emocion == "happy" and ojos_resultado['sonrisa_genuina']:
        return 0.1, "✅ Sonrisa genuina detectada (Duchenne)"
    
    ajustes = {
        "angry": (0.3, "⚠️ Cejas fruncidas (enojo ≠ depresión)"),
        "surprise": (0.2, "👀 Ojos muy abiertos (sorpresa)"),
        "disgust": (0.4, "🤢 Nariz arrugada (asco)"),
        "fear": (0.5, "😨 Músculos tensos (miedo)")
    }
    
    return ajustes.get(emocion, (1.0, ""))

def procesar_frame(frame):
    try:
        # Análisis de emociones
        try:
            resultado = DeepFace.analyze(img_path=frame, 
                                      actions=["emotion"], 
                                      enforce_detection=False,
                                      detector_backend='mtcnn')
            emocion = resultado[0]['dominant_emotion']
            emociones = resultado[0]['emotion']
        except Exception as e:
            print(f"Error en análisis de emociones: {str(e)}")
            emocion = "neutral"
            emociones = {'sad': 0, 'neutral': 100}

        # Análisis ocular
        ojos_resultado = analizar_ojos_depresivos(frame)
        
        if not ojos_resultado:
            return None, None, None, None, None
        
        # Aplicar filtros
        factor_ajuste, mensaje = filtrar_emociones(emocion, emociones, ojos_resultado)
        
        # Cálculo final del porcentaje
        porcentaje = min(100, (
            emociones['sad'] * 0.4 * factor_ajuste + 
            emociones['neutral'] * 0.2 + 
            ojos_resultado['puntaje_total'] * 0.4 * factor_ajuste
        ))

        return emocion, emociones, ojos_resultado, porcentaje, mensaje
    
    except Exception as e:
        print(f"Error procesando frame: {str(e)}")
        return None, None, None, None, None

def guardar_registro(emocion, porcentaje, ojos_resultado):
    registro = {
        'fecha': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'puntaje': porcentaje,
        'emocion_dominante': emocion,
        'cejas': ojos_resultado['cejas_estado'],
        'parpados': ojos_resultado['parpados_estado'],
        'comisuras': ojos_resultado['comisuras_estado'],
        'nota': "Detección automática"
    }
    
    df = pd.DataFrame([registro])
    if os.path.exists(archivo_emociones):
        df.to_csv(archivo_emociones, mode='a', header=False, index=False)
    else:
        df.to_csv(archivo_emociones, index=False)

# ========== INTERFAZ STREAMLIT ==========
st.header("Monitoreo Automático de Expresiones")

# Inicializar variables de estado
if 'running' not in st.session_state:
    st.session_state.running = False
if 'ultima_deteccion' not in st.session_state:
    st.session_state.ultima_deteccion = None
if 'notificacion' not in st.session_state:
    st.session_state.notificacion = None
if 'ultimo_registro' not in st.session_state:
    st.session_state.ultimo_registro = None

def toggle_camera():
    st.session_state.running = not st.session_state.running

# Controles
col1, col2 = st.columns(2)
with col1:
    st.button("Iniciar/Detener Monitoreo", on_click=toggle_camera)
with col2:
    umbral_tristeza = st.slider("Umbral de tristeza para registro", 40, 90, 60)

# Contenedores
video_placeholder = st.empty()
results_placeholder = st.empty()
notification_placeholder = st.empty()

# Inicializar cámara
if st.session_state.running:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
else:
    if 'cap' in locals():
        cap.release()

# Procesamiento de frames
if st.session_state.running:
    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("No se pudo acceder a la cámara")
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # Procesar frame cada 2 segundos (para no sobrecargar el sistema)
        if time.time() - st.session_state.get('ultimo_procesamiento', 0) > 2:
            emocion, emociones, ojos_resultado, porcentaje, mensaje = procesar_frame(frame)
            st.session_state.ultimo_procesamiento = time.time()
            
            if porcentaje is not None:
                # Mostrar resultados actuales
                if porcentaje > umbral_tristeza:
                    box_class = "danger-box" if porcentaje > 70 else "warning-box"
                    evaluacion = "🔴 Posible tristeza" if porcentaje > 70 else "🟠 Indicios de tristeza"
                    
                    results_placeholder.markdown(f"""
                    <div class="{box_class}">
                        <h3>{evaluacion}</h3>
                        <p><strong>Puntaje:</strong> {porcentaje:.1f}%</p>
                        <p><strong>Emoción dominante:</strong> {emocion.upper()}</p>
                        <p><small>Última actualización: {datetime.now().strftime("%H:%M:%S")}</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Guardar registro si supera el umbral
                    if porcentaje > umbral_tristeza and (st.session_state.ultimo_registro is None or 
                                                       time.time() - st.session_state.ultimo_registro > 10):
                        guardar_registro(emocion, porcentaje, ojos_resultado)
                        st.session_state.ultimo_registro = time.time()
                        st.session_state.notificacion = f"Registro guardado: {porcentaje:.1f}% de tristeza"
                
                # Mostrar notificación temporal
                if st.session_state.notificacion and time.time() - st.session_state.ultimo_registro < 3:
                    notification_placeholder.markdown(
                        f'<div class="notification">{st.session_state.notificacion}</div>', 
                        unsafe_allow_html=True
                    )
                else:
                    notification_placeholder.empty()
                    st.session_state.notificacion = None
        
        # Pequeña pausa para no saturar
        cv2.waitKey(100)
        
    # Liberar cámara al detener
    if 'cap' in locals():
        cap.release()
else:
    video_placeholder.info("Presiona 'Iniciar/Detener Monitoreo' para comenzar la detección automática")

# Mostrar historial si existe
if os.path.exists(archivo_emociones):
    st.header("Registros Automáticos")
    df = pd.read_csv(archivo_emociones)
    
    # Filtrar solo detecciones relevantes
    df_relevante = df[df['puntaje'] >= umbral_tristeza]
    
    if not df_relevante.empty:
        st.dataframe(df_relevante.sort_values('fecha', ascending=False))
        
        # Gráfico de tendencias
        st.subheader("Tendencia de detecciones")
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['hora'] = df['fecha'].dt.strftime('%H:%M')
        st.line_chart(df.set_index('fecha')['puntaje'])
    else:
        st.info("No hay registros relevantes según el umbral actual")