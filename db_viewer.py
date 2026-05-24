import os
import sys
import chromadb

def run_db_viewer():
    # The actual database path for Chronos is ./chroma_db
    db_path = "./chroma_db"
    
    if not os.path.exists(db_path):
        # Fallback to check parent or sibling if not running from chronos folder
        if os.path.exists("../chronos/chroma_db"):
            db_path = "../chronos/chroma_db"
        elif os.path.exists("./chronos/chroma_db"):
            db_path = "./chronos/chroma_db"
            
    print("=" * 60)
    print("           CHRONOS VECTOR DATABASE INSPECTOR")
    print("=" * 60)
    print(f"Connecting to ChromaDB at: '{db_path}'")
    
    try:
        client = chromadb.PersistentClient(path=db_path)
        collections = client.list_collections()
    except Exception as e:
        print(f"\n[!] Error connecting to ChromaDB: {e}")
        print("[*] Please make sure you run this script from the 'chronos' root directory.")
        sys.exit(1)
        
    print(f"\nFound {len(collections)} collections in database:")
    for idx, col in enumerate(collections, 1):
        print(f" [{idx}] '{col.name}' (Contains {col.count()} items)")
        
    for col in collections:
        print("\n" + "=" * 60)
        print(f"COLLECTION: '{col.name}' (Total items: {col.count()})")
        print("=" * 60)
        
        count = col.count()
        if count == 0:
            print("Collection is empty.")
            continue
            
        if col.name == "security_events":
            print("\n--- All Stored Cognitive Alerts & System Logs ---")
            # Retrieve all documents and metadatas, excluding embeddings to keep print output clean
            results = col.get(include=["documents", "metadatas"])
            
            ids = results.get('ids', [])
            documents = results.get('documents', [])
            metadatas = results.get('metadatas', [])
            
            # Sort by timestamp for chronological order
            events = []
            for item_id, doc, meta in zip(ids, documents, metadatas):
                ts = meta.get('timestamp', 0) if meta else 0
                events.append((ts, item_id, doc))
            events.sort(key=lambda x: x[0])
            
            for i, (ts, item_id, doc) in enumerate(events):
                import datetime
                dt = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S') if ts else "N/A"
                print(f"[{dt}] ID: {item_id}")
                print(f"  Event: {doc}\n")
                
        elif col.name == "person_reid":
            print("\n--- Tracked Persistent Subject IDs ---")
            results = col.get(include=["metadatas"])
            ids = results.get('ids', [])
            metadatas = results.get('metadatas', [])
            
            # Print subjects
            print(f"Registered Subject List ({len(ids)} subjects):")
            for i, (subj_id, meta) in enumerate(zip(ids, metadatas)):
                print(f"  Subject Name: {subj_id} | System Tag: {meta.get('system', 'N/A')}")
        else:
            # General fallback for any other collections
            print("\n--- Raw Data Peek ---")
            print(col.peek(5))

if __name__ == "__main__":
    run_db_viewer()
