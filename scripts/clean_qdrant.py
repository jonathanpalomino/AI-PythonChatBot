import requests

QDRANT_URL = "http://localhost:6333"
KEEP_COLLECTION = "documents"  # ← CAMBIA ESTO


def delete_all_except_one():
    try:
        # Obtener todas las colecciones
        response = requests.get(f"{QDRANT_URL}/collections")
        response.raise_for_status()

        collections = response.json()["result"]["collections"]
        collection_names = [c["name"] for c in collections]

        print(f"\nColecciones encontradas: {len(collection_names)}")
        for name in collection_names:
            print(f"  - {name}")

        # Filtrar la que queremos mantener
        to_delete = [name for name in collection_names if name != KEEP_COLLECTION]

        if not to_delete:
            print(f"\n✓ No hay colecciones para borrar.")
            print(f"✓ Solo existe: {KEEP_COLLECTION}")
            return

        print(f"\n⚠️  Se borrarán {len(to_delete)} colecciones:")
        for name in to_delete:
            print(f"  - {name}")
        print(f"\n✓ Se mantendrá: {KEEP_COLLECTION}")

        # Confirmación
        confirm = input("\n¿Continuar? (escribe 'si' para confirmar): ")
        if confirm.lower() != 'si':
            print("❌ Operación cancelada")
            return

        # Borrar cada colección
        print("\nBorrando colecciones...")
        for name in to_delete:
            print(f"  Borrando '{name}'...", end=" ")
            delete_response = requests.delete(f"{QDRANT_URL}/collections/{name}")

            if delete_response.status_code == 200:
                print("✓")
            else:
                print(f"✗ Error: {delete_response.text}")

        print(f"\n✅ Completado. Solo queda: {KEEP_COLLECTION}")

    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar a Qdrant.")
        print("   Asegúrate de que Qdrant esté corriendo en http://localhost:6333")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


if __name__ == "__main__":
    delete_all_except_one()
