import weaviate
import os
from dotenv import load_dotenv

# --- Configuration ---
WEAVIATE_CLASS_NAME = "InsurancePolicy"

def main():
    """Connects to Weaviate and deletes the specified class using V4 syntax."""
    load_dotenv()
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if not weaviate_url or not weaviate_api_key:
        print("🛑 ERROR: WEAVIATE_URL and WEAVIATE_API_KEY must be set in the .env file.")
        return

    client = None  # Initialize client to None
    try:
        print(f"Connecting to Weaviate at {weaviate_url}...")
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate.auth.AuthApiKey(weaviate_api_key)
        )
        print("✅ Successfully connected to Weaviate.")

        # --- THIS IS THE CORRECTED V4 SYNTAX ---
        # Instead of client.schema, we use client.collections
        if client.collections.exists(WEAVIATE_CLASS_NAME):
            print(f"Found existing collection '{WEAVIATE_CLASS_NAME}'. Deleting it now...")
            # The new method to delete a collection
            client.collections.delete(WEAVIATE_CLASS_NAME)
            print(f"✅ Collection '{WEAVIATE_CLASS_NAME}' has been deleted.")
        else:
            print(f"✅ Collection '{WEAVIATE_CLASS_NAME}' does not exist. No action needed.")
        # --- END OF CORRECTION ---
            
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Check if client was successfully created and is connected before closing
        if client and client.is_connected():
            client.close()
            print("Connection to Weaviate closed.")

if __name__ == "__main__":
    main()