Face Recognition API

Servicio backend para captura y verificación biométrica facial, desarrollado en FastAPI, con almacenamiento de vectores faciales en MongoDB.
Permite la captura guiada de tres poses (frontal, arriba y abajo), la extracción de vectores faciales usando face_recognition y la verificación posterior contra registros almacenados.

Características principales

API basada en FastAPI y Uvicorn.

Captura facial en tres poses: front, up, down.

Estimación de orientación facial (pitch/yaw) mediante landmarks y OpenCV.

Generación de vectores de 128 dimensiones mediante face_recognition.

Persistencia en MongoDB con motor asíncrono motor.

Validación, reseteo y depuración del estado de captura.

Docker Compose con servicios integrados de Mongo y Mongo Express.

Compatible con entornos arm64 y amd64 (por ejemplo, Apple Silicon).

Estructura del proyecto
fastapi-face/
│
├── main.py                 # API principal (endpoints de captura, registro y verificación)
├── db.py                   # Inicialización y cierre de conexión a MongoDB
├── environment.yml          # Dependencias Conda (entorno "fr")
├── Dockerfile               # Imagen del servicio backend
├── docker-compose.yml       # Servicios (API, MongoDB, Mongo Express)
├── data/                    # Carpeta persistente de datos locales
└── README.md

Requisitos previos

Docker y Docker Compose instalados.

Puerto 8000 libre para el backend.

Puerto 27017 (Mongo) y 8081 (Mongo Express) disponibles.



Construir y levantar los contenedores:

docker compose build --no-cache
docker compose up -d


Verificar logs del servicio:

docker compose logs -f api


Acceder al panel de Mongo Express (opcional):

http://localhost:8081


Endpoints principales
Sistema
Método	Endpoint	Descripción
GET	/	Estado general del servicio
GET	/health	Verifica el estado del backend
GET	/debug_state	Muestra el estado de poses y progreso actual
Captura guiada
Método	Endpoint	Descripción
POST	/analyze_frame	Procesa un frame (pose actual, pitch/yaw, vector si aplica)
POST	/reset	Reinicia el estado de captura completo
POST	/reset_pose	Reinicia una pose específica
GET	/progress	Devuelve avance de poses capturadas
POST	/finalize_registration	Calcula y guarda el vector promedio del usuario
Verificación
Método	Endpoint	Descripción
POST	/verify_frame	Verifica una imagen contra un usuario registrado
Gestión de usuarios
Método	Endpoint	Descripción
GET	/users	Lista usuarios registrados
GET	/users/{usuarioID}	Devuelve los datos de un usuario
GET	/check_user/{usuarioID}	Verifica si existe un vector guardado

Ejemplo de flujo de registro

El cliente (front) envía imágenes a /analyze_frame para las tres poses.

Cada pose válida genera un vector facial (128 valores).

Al completar las tres poses, se ejecuta /finalize_registration con el usuarioID.

El backend promedia los tres vectores normalizados y lo almacena en MongoDB.

Para verificación futura, /verify_frame compara un nuevo vector con el almacenado.

Dependencias principales

fastapi

uvicorn[standard]

opencv-python-headless

face-recognition

motor

numpy

Pillow

Ejemplo de prueba en Postman

Importar la colección face-api.postman_collection.json.

Probar en orden:

POST /analyze_frame (enviar campo imagen)

GET /progress

POST /finalize_registration con usuarioID

POST /verify_frame para comparar rostro nuevo.

Licencia

Este proyecto está distribuido bajo la licencia MIT.
Puedes usarlo, modificarlo y distribuirlo libremente con los créditos correspondientes.
