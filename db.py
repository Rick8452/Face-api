# db.py
import os
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI

mongo_client = None


async def init_mongo(app: FastAPI):

    uri = os.getenv("MONGO_URI")
    dbname = os.getenv("MONGO_DB", "face")
    colname = os.getenv("MONGO_COL", "users")

    if not uri:
        print("[Mongo] MONGO_URI no definido. Modo archivos activo.")
        return

    global mongo_client
    mongo_client = AsyncIOMotorClient(uri, uuidRepresentation="standard")
    db = mongo_client[dbname]
    col = db[colname]

    await col.create_index("usuarioID", unique=True)
    app.state.mongo_col = col
    print(f"[Mongo] Conectado: {uri} (db={dbname}, col={colname})")


async def close_mongo():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        mongo_client = None
        print("[Mongo] Conexión cerrada.")
