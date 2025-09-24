from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, face_recognition, time, json, os
from typing import Optional
from PIL import Image, ImageOps
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ajusta si quieres restringir
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Memoria de captura guiada -----------------
progress = {"front": None, "up": None, "down": None}
timestamps = {"front": None, "up": None, "down": None}
current_pose = None
HOLD_TIME = 2  # s

# ----------------- Carpeta para vectores persistentes -----------------
DATA_DIR = "data/users"
os.makedirs(DATA_DIR, exist_ok=True)

def _save_user_vector(user_id: str, vector: list[float]):
    path = os.path.join(DATA_DIR, f"{user_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id, "vector": vector}, f, ensure_ascii=False)

def _load_user_vector(user_id: str) -> Optional[list[float]]:
    path = os.path.join(DATA_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("vector")

# ----------------- Utilidades de imagen -----------------
def _rotationMatrixToEulerAngles(R: np.ndarray):
    sy = np.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # roll
        y = np.arctan2(-R[2,0], sy)     # pitch
        z = np.arctan2(R[1,0], R[0,0])  # yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1]); y = np.arctan2(-R[2,0], sy); z = 0
    return x, y, z

def _estimate_pitch_yaw(img_bgr: np.ndarray):
    print("=== INICIO ESTIMACIÓN POSE ===")
    
    # Probar tanto la imagen original como la volteada
    img_bgr_flipped = cv2.flip(img_bgr, 1)
    
    # Procesar imagen volteada (que suele dar mejores resultados)
    img_rgb = cv2.cvtColor(img_bgr_flipped, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img_rgb, model="hog")
    print(f"Caras detectadas: {len(boxes)}")
    if not boxes:
        print(" No se detectaron caras en imagen volteada, intentando con original...")
        # Intentar con imagen original
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(img_rgb, model="hog")
        if not boxes:
            print(" No se detectaron caras en ninguna imagen")
            return None, None
    
    landmarks_list = face_recognition.face_landmarks(img_rgb)
    if not landmarks_list: 
        print(" No se pudieron detectar landmarks")
        return None, None
        
    lm = landmarks_list[0]
    print(" Landmarks detectados correctamente")

    try:
        nose_tip = np.mean(lm["nose_tip"], axis=0)
        chin = lm["chin"][8]
        left_eye_corner = lm["left_eye"][0]
        right_eye_corner = lm["right_eye"][3]

        print(f" Punta nariz: {nose_tip}")
        print(f" Mentón: {chin}")
        print(f" Esquina ojo izquierdo: {left_eye_corner}")
        print(f" Esquina ojo derecho: {right_eye_corner}")

        # Verificar si los landmarks están en la posición correcta
        if left_eye_corner[0] > right_eye_corner[0]:
            print(" ¡Los ojos están invertidos! Usando imagen original...")
            # Usar imagen original sin voltear
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            landmarks_list = face_recognition.face_landmarks(img_rgb)
            if not landmarks_list:
                return None, None
            lm = landmarks_list[0]
            
            # Recalcular puntos con imagen original
            nose_tip = np.mean(lm["nose_tip"], axis=0)
            chin = lm["chin"][8]
            left_eye_corner = lm["left_eye"][0]
            right_eye_corner = lm["right_eye"][3]

        mouth_points = np.array(lm["top_lip"] + lm["bottom_lip"])
        left_mouth_corner = mouth_points[np.argmin(mouth_points[:,0])]
        right_mouth_corner = mouth_points[np.argmax(mouth_points[:,0])]
        image_points = np.array([
            nose_tip, chin, left_eye_corner, right_eye_corner,
            left_mouth_corner, right_mouth_corner
        ], dtype=np.float64)
        
    except Exception as e:
        print(f" Error al extraer landmarks: {e}")
        return None, None
        
    model_points = np.array([
        (0.0,0.0,0.0),(0.0,-330.0,-65.0),(-225.0,170.0,-135.0),(225.0,170.0,-135.0),
        (-150.0,-150.0,-125.0),(150.0,-150.0,-125.0)
    ], dtype=np.float64)
    
    h,w = img_rgb.shape[:2]
    print(f"Dimensiones imagen: {w}x{h}")
    focal_length = w
    camera_matrix = np.array([
        [focal_length, 0, w/2],
        [0, focal_length, h/2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    dist_coeffs = np.zeros((4,1))
    success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    if not success: 
        print("solvePnP falló")
        return None, None
        
    R,_ = cv2.Rodrigues(rvec)
    roll, pitch, yaw = _rotationMatrixToEulerAngles(R)
    pitch_deg = float(np.degrees(pitch))
    yaw_deg = float(np.degrees(yaw))
    roll_deg = float(np.degrees(roll))
    
    print(f" Ángulos calculados - Roll: {roll_deg:.2f}°, Pitch: {pitch_deg:.2f}°, Yaw: {yaw_deg:.2f}°")

    return pitch_deg, yaw_deg

def _correct_yaw_angle(yaw: float) -> float:
    """Corrige ángulos de yaw fuera de rango normal"""
    if yaw is None:
        return 0.0
    
    yaw_normalized = yaw % 360
    if yaw_normalized > 180:
        yaw_normalized -= 360
    
    if abs(yaw_normalized) > 90:
        if yaw_normalized > 0:
            yaw_normalized -= 180
        else:
            yaw_normalized += 180
    
    return yaw_normalized


def _classify_pose(pitch: float, yaw: float) -> str | None:
    print(f" Clasificando pose - Pitch: {pitch:.2f}, Yaw: {yaw:.2f}")
    
    # Corregir yaw primero
    yaw_corrected = _correct_yaw_angle(yaw)
    print(f" Yaw original: {yaw:.2f}°, Yaw corregido: {yaw_corrected:.2f}°")
    
    YAW_THRESHOLD = 25  
    PITCH_THRESHOLD = 15  
    
    
    is_tilt_pose = abs(pitch) >= PITCH_THRESHOLD
    yaw_threshold = YAW_THRESHOLD * (1.5 if is_tilt_pose else 1) 
    
    if abs(yaw_corrected) > yaw_threshold:
        print(f" Yaw corregido excede umbral: {abs(yaw_corrected):.2f} > {yaw_threshold}")
        return None
        
    if pitch <= -PITCH_THRESHOLD:
        print(" Pose clasificada: up")
        return "up"
        
    if pitch >= PITCH_THRESHOLD:
        print(" Pose clasificada: down")
        return "down"
        
    print(" Pose clasificada: front")
    return "front"

def _load_image_as_bgr_from_bytes(bytes_data: bytes, max_width: int = 900) -> np.ndarray | None:
    try:
        pil = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        pil = ImageOps.exif_transpose(pil)
        if pil.width > max_width:
            ratio = max_width / pil.width
            pil = pil.resize((max_width, int(pil.height * ratio)))
        arr = np.array(pil)
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except Exception:
        nparr = np.frombuffer(bytes_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def _get_face_vector(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(img_rgb, model="hog")
    if not boxes: 
        print(" No se detectaron caras para extraer vector")
        return None
    enc = face_recognition.face_encodings(img_rgb, boxes)
    if not enc: 
        print(" No se pudieron extraer encodings faciales")
        return None
    print(f" Vector facial extraído. Longitud: {len(enc[0])}")
    return enc[0]  # numpy array (128,)

def _to_list(vec: np.ndarray) -> list[float]:
    return [float(x) for x in vec.tolist()]

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v);  return v if n==0 else v/n

# ----------------- Endpoints de captura guiada -----------------
@app.post("/analyze_frame")
async def analyze_frame(file: Optional[UploadFile] = File(None), imagen: Optional[UploadFile] = File(None)):
    """Acepta 'file' o 'imagen' como campo multipart."""
    print("=== INICIO ANALYZE_FRAME ===")
    global current_pose
    upload = file or imagen
    if upload is None:
        print(" Error: No se recibió archivo")
        return JSONResponse(content={"status": "no_file"}, status_code=400)
    
    contenido = await upload.read()
    if not contenido:
        print("Error: Archivo vacío")
        return JSONResponse(content={"status": "no_file"}, status_code=400)

    img = _load_image_as_bgr_from_bytes(contenido)
    if img is None:
        return JSONResponse(content={"status": "bad_image"}, status_code=400)
    print(f" Tamaño archivo recibido: {len(contenido)} bytes")

    pitch, yaw = _estimate_pitch_yaw(img)
    if pitch is None:
        return {"status": "no_face"}

    # Calcular yaw corregido para la respuesta
    yaw_corrected = _correct_yaw_angle(yaw) if yaw is not None else yaw
    
    pose = _classify_pose(pitch, yaw)
    now = time.time()

    if pose is None:
        current_pose = None
        for k in timestamps: timestamps[k] = None
        # Enviar el YAW CORREGIDO en la respuesta
        return {"status": "wrong_direction", "yaw": round(yaw_corrected, 2)}

    # VERIFICAR SI LA POSE YA ESTÁ CAPTURADA
    if progress[pose] is not None:
        return {"status": "already_done", "pose": pose}

    # SI ES UNA NUEVA POSE, RESETEAR ESTADO
    if current_pose != pose:
        current_pose = pose
        timestamps[pose] = now
        return {"status": "waiting", "pose": pose, "message": f"Mantén {pose} {HOLD_TIME}s"}

    # VERIFICAR TIEMPO DE MANTENIMIENTO
    elapsed = now - (timestamps[pose] or now)
    if elapsed >= HOLD_TIME:
        vec = _get_face_vector(img)
        if vec is None:
            return {"status": "no_face_vector"}
        progress[pose] = _to_list(vec)
        current_pose = None
        timestamps[pose] = None
        print(f"Pose {pose} capturada exitosamente. Vector length: {len(progress[pose])}")
        return {"status": "captured", "pose": pose, "vector": progress[pose]}
    else:
        return {"status": "waiting", "pose": pose, "elapsed": round(elapsed,2)}

@app.get("/progress")
def get_progress():
    done = sum(1 for k in ["front","up","down"] if progress[k] is not None)
    return {"progress": done/3, "vectors": progress}

@app.post("/reset")
def reset():
    global current_pose
    for k in list(progress.keys()): progress[k] = None
    for k in list(timestamps.keys()): timestamps[k] = None
    current_pose = None
    print("Estado del servicio reseteado completamente")
    return {"status": "reset_ok"}

@app.post("/reset_pose")
def reset_pose(pose: str = Form(...)):
    """Resetear una pose específica"""
    global current_pose
    if pose in progress:
        progress[pose] = None
        timestamps[pose] = None
        if current_pose == pose:
            current_pose = None
        print(f"Pose {pose} reseteada")
        return {"status": f"reset_ok_{pose}"}
    else:
        return JSONResponse(content={"status": "invalid_pose"}, status_code=400)

@app.get("/debug_state")
def debug_state():
    """Endpoint para debug del estado actual"""
    vector_info = {}
    for k, v in progress.items():
        if v is not None:
            vector_info[k] = f"length: {len(v)}"
        else:
            vector_info[k] = "None"
    
    return {
        "current_pose": current_pose,
        "progress": vector_info,
        "timestamps": timestamps,
        "hold_time": HOLD_TIME
    }

# ----------------- Finalizar registro (persiste vector del usuario) -----------------
@app.post("/finalize_registration")
async def finalize_registration(user_id: str = Form(...)):
    # Verificar que tengamos 3 vectores capturados y que no estén vacíos
    vectors_present = {}
    for k in ("front","up","down"):
        vectors_present[k] = progress[k] is not None and len(progress[k]) == 128
        print(f"Pose {k}: {'VÁLIDA' if vectors_present[k] else 'INVÁLIDA'}")

    if not all(vectors_present.values()):
        missing = [k for k, v in vectors_present.items() if not v]
        return JSONResponse(
            content={"status": "incomplete", "missing": missing}, 
            status_code=400
        )

    v_front = np.array(progress["front"], dtype=np.float32)
    v_up    = np.array(progress["up"], dtype=np.float32)
    v_down  = np.array(progress["down"], dtype=np.float32)

    # Promedio de vectores normalizados (robusto a pequeñas variaciones)
    mean_vec = _normalize(_normalize(v_front) + _normalize(v_up) + _normalize(v_down))
    _save_user_vector(user_id, _to_list(mean_vec))

    # Limpia progreso en memoria para el siguiente usuario
    for k in list(progress.keys()): progress[k] = None
    for k in list(timestamps.keys()): timestamps[k] = None
    current_pose = None

    print(f"Registro finalizado para usuario {user_id}")
    return {"status": "saved", "user_id": user_id}

# ----------------- Verificación (comparar contra usuario guardado) -----------------
THRESHOLD = 0.35  # ajusta según pruebas (0.45-0.6 típico HOG)

@app.post("/verify_frame")
async def verify_frame(user_id: str = Form(...), file: Optional[UploadFile] = File(None), imagen: Optional[UploadFile] = File(None)):
    """Verificación biométrica con mejor manejo de errores"""
    try:
        print(f"=== INICIO VERIFICACIÓN PARA USUARIO {user_id} ===")
        
        upload = file or imagen
        if upload is None:
            print("Error: No se recibió archivo")
            return JSONResponse(content={"status": "no_file"}, status_code=400)
        
        contenido = await upload.read()
        if not contenido:
            print("Error: Archivo vacío")
            return JSONResponse(content={"status": "no_file"}, status_code=400)

        # Cargar vector de referencia
        print("Cargando vector de referencia...")
        ref = _load_user_vector(user_id)
        if ref is None:
            print(f"No se encontró vector para usuario {user_id}")
            # Listar archivos disponibles para debug
            import glob, os
            files = glob.glob(os.path.join(DATA_DIR, "*.json"))
            print(f"Archivos disponibles: {[os.path.basename(f) for f in files]}")
            return JSONResponse(content={"status": "no_reference"}, status_code=404)

        print("Vector de referencia cargado exitosamente")

        img = _load_image_as_bgr_from_bytes(contenido)
        if img is None:
            print("Error: No se pudo decodificar la imagen")
            return JSONResponse(content={"status": "bad_image"}, status_code=400)

        vec = _get_face_vector(img)
        if vec is None:
            print("Error: No se pudo extraer vector facial de la imagen")
            return {"status": "no_face"}

        # Calcular distancia
        dist = float(face_recognition.face_distance([np.array(ref, dtype=np.float32)], vec)[0])
        match = bool(dist < THRESHOLD)
        
        print(f"Verificación completada. Distancia: {dist:.4f}, Match: {match}")
        
        return {
            "status": "ok", 
            "user_id": user_id, 
            "distance": dist, 
            "threshold": THRESHOLD, 
            "match": match
        }
        
    except Exception as e:
        print(f"Error en verificación: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

# ----------------- Gestión de usuarios -----------------
@app.get("/users")
def list_users():
    import glob, os, json
    files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    return [{"user_id": os.path.splitext(os.path.basename(p))[0]} for p in files]

@app.get("/users/{user_id}")
def get_user(user_id: str):
    import os, json
    path = os.path.join(DATA_DIR, f"{user_id}.json")
    if not os.path.exists(path):
        return JSONResponse(content={"status": "not_found"}, status_code=404)
    return json.load(open(path, "r", encoding="utf-8"))

# ----------------- Endpoints de sistema -----------------
@app.get("/")
def root():
    return {"status": "ok", "service": "face_recognition_api"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "timestamp": time.time(),
        "current_pose": current_pose,
        "poses_captured": sum(1 for k in progress if progress[k] is not None)
    }

@app.post("/save_vector")
async def save_vector(user_id: str = Form(...), vector: str = Form(...)):
    """Endpoint para guardar un vector directamente desde NestJS"""
    try:
        print(f"=== GUARDANDO VECTOR PARA USUARIO {user_id} ===")
        
        # Convertir el string JSON a lista de floats
        vector_list = json.loads(vector)
        print(f"Vector recibido. Longitud: {len(vector_list)}")
        
        # Validar que sea un vector válido
        if not isinstance(vector_list, list) or len(vector_list) != 128:
            return JSONResponse(
                content={"status": "error", "message": "Vector inválido"},
                status_code=400
            )
        
        # Guardar el vector
        _save_user_vector(user_id, vector_list)
        print(f"Vector guardado exitosamente para usuario {user_id}")
        
        return {"status": "saved", "user_id": user_id}
        
    except json.JSONDecodeError:
        return JSONResponse(
            content={"status": "error", "message": "Vector en formato JSON inválido"},
            status_code=400
        )
    except Exception as e:
        print(f"Error al guardar vector: {str(e)}")
        return JSONResponse(
            content={"status": "error", "message": str(e)},
            status_code=500
        )

@app.get("/check_user/{user_id}")
def check_user(user_id: str):
    """Endpoint para verificar si un usuario tiene vector guardado"""
    vector = _load_user_vector(user_id)
    if vector is None:
        return {"exists": False, "message": "Usuario no encontrado"}
    
    return {
        "exists": True, 
        "user_id": user_id, 
        "vector_length": len(vector) if vector else 0
    }

@app.post("/debug_pose")
async def debug_pose(file: UploadFile = File(...)):
    """Endpoint para diagnóstico detallado de la detección de pose."""
    try:
        contenido = await file.read()
        print("=== MODO DIAGNÓSTICO ===")
        
        img = _load_image_as_bgr_from_bytes(contenido)
        if img is None:
            return {"error": "No se pudo decodificar la imagen"}
            
        print(f" Imagen decodificada: {img.shape}")
        
        # Procesar imagen original
        pitch1, yaw1 = _estimate_pitch_yaw(img)
        
        # Procesar imagen volteada
        img_flipped = cv2.flip(img, 1)
        pitch2, yaw2 = _estimate_pitch_yaw(img_flipped)
        
        # Clasificar poses
        pose1 = _classify_pose(pitch1, yaw1) if pitch1 is not None else None
        pose2 = _classify_pose(pitch2, yaw2) if pitch2 is not None else None
        
        # Verificar detección de vector facial
        vec_original = _get_face_vector(img)
        vec_flipped = _get_face_vector(img_flipped)
        
        return {
            "original": {
                "pitch": pitch1, 
                "yaw": yaw1, 
                "pose": pose1,
                "vector_detected": vec_original is not None
            },
            "flipped": {
                "pitch": pitch2, 
                "yaw": yaw2, 
                "pose": pose2,
                "vector_detected": vec_flipped is not None
            },
            "current_service_state": {
                "current_pose": current_pose,
                "progress": {k: v is not None for k, v in progress.items()}
            }
        }
        
    except Exception as e:
        print(f" Error en debug_pose: {str(e)}")
        return {"error": str(e)}
    
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)