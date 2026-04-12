'''
Author: Armand Meijers
Date: 02/04/2026
Description: Ingestion pipeline code, chunking pdf files and embedding chunks
'''

import os, json, faiss, pymupdf
from . import helper
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_JSON_PATH = "data/processed/metadata/chunk_metadata.json"

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

#helper functions
def load_or_create_index(vector_path: str, chunk_json_path: str):
    """
    Safely loads a FAISS index. If the index is missing, empty, or corrupted,
    it rebuilds it from chunk metadata.
    """

    if os.path.exists(vector_path) and os.path.getsize(vector_path) > 0:
        try:
            print("[LOG] Loading existing faiss index...")
            return faiss.read_index(vector_path)
        except Exception as e:
            print(f"[WARN] Faiss index error {e}")

    else:
        print("[LOG] Faiss index missing. downloading and re-embeddings...")
        helper.path_checker_creator(vector_path)
        


    # rebuild index
    index, _ = embedding_text(chunk_json_path, vector_path)
    return index


def chunking_files_pdf(DOCUMENT_PATH: str) -> list[dict]:
    """
    Reads all PDFs in DOCUMENTS_PATH, splits pages into chunks,
    and returns a list of chunk objects with metadata.

    Args:
        DOCUMENT_PATH (str):    path to document folder (e.g., data/raw/papers) 

    Returns:
        List[Dict]: Each dict has "text" and "metadata"  with valyes filename, page, chunk_index
    """

    has_files = any(os.path.isfile(os.path.join(DOCUMENT_PATH, f)) for f in os.listdir(DOCUMENT_PATH))
    if not has_files:
        print("[ERROR] run download script in src first!")
        return False

    #chunk tokens
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    #chunk arrays
    chunk_metadata = []
    
    #text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )   

    for file in os.listdir(DOCUMENT_PATH):

        #checks if file is a pdf
        if not file.endswith(".pdf"):
            continue

        page_index = 0
        chunk_index = 0

        file_path = os.path.join(DOCUMENT_PATH, file)

        #error handling
        try:
            doc = pymupdf.open(file_path)
        except Exception as e:
            print(f"Skipping {file} Error: {e}")
            continue

        try:
            #loop through all pages in doc
            for page in doc:
                #checks if doc page has text (can have no text only images ect)
                page_text = page.get_text()
                page_index += 1

                if page_text.strip() == "":
                    continue

                #chunking of page
                chunks = text_splitter.split_text(page_text)

                #loops through each chunk
                for chunk in chunks:
                    chunk_index += 1
                    
                    #append metadata to array
                    chunk_metadata.append({
                        "id": f"{file}_{page_index}_{chunk_index}",
                        "text": chunk,
                        "meta": {
                            "filename": file,
                            "page": page_index,
                            "chunk_index": chunk_index,
                        }
                    })

        #errpr handling
        except Exception as e:
            print(f"Error: {file}: {e}")

        finally:
            #close doc
            doc.close()

    #write chunk metadata
    with open(CHUNK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_metadata, f, indent=2, ensure_ascii=False)
        
    return chunk_metadata

def append_chunks_pdf(new_files: list[str], json_path: str) -> list[dict]:
    """
    Append chunks from new PDFs to an existing chunk metadata JSON.

    Args:
        new_files (list[str]): list of new PDF paths
        json_path (str): path to JSON file to append data to

    return:
        chunk_metadata(List[Dict]): New data that got appended
    """
    
    #chunk tokens
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    new_chunks = []

    #load existing metadata if exists
    if os.path.exists(json_path) and os.path.getsize(json_path) > 0:
        with open(json_path, "r", encoding="utf-8") as f:
            chunk_data = json.load(f)
    else:
        chunk_data = []

    chunk_index = 0
    if chunk_data:
        chunk_index = max(chunk["meta"]["chunk_index"] for chunk in chunk_data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
    )

    for path in new_files:
        page_index = 0
        filename = os.path.basename(path)  # remove full path, keep only file name

        try:
            doc = pymupdf.open(path)
        except Exception as e:
            print(f"[WARN] Skipping {path} Error: {e}")
            continue

        try:
            for page in doc:
                page_index += 1
                page_text = page.get_text()
                
                if not page_text.strip(): 
                    continue

                chunks = text_splitter.split_text(page_text)

                for chunk in chunks:
                    chunk_index += 1
                    
                    chunk_data.append({
                        "id": f"{filename}_{page_index}_{chunk_index}",
                        "text": chunk,
                        "meta": {
                            "filename": filename,
                            "page": page_index,
                            "chunk_index": chunk_index,
                        }
                    })

                    chunk_data.append({chunk_data})
                    chunk_data.append(chunk_data) 
                    new_chunks.append(chunk_data)

                

        except Exception as e:
            print(f"Error: {filename}: {e}")

        finally:
            doc.close()

    #save back to json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(chunk_data, f, indent=2, ensure_ascii=False)

    print(f"[LOG] Appended {len(new_chunks)} chunks from {len(new_files)} files")

    return new_chunks



def embedding_text(datafile_path: str, vector_path: str) -> str:
    """
    Takes metadata JSON path (with chunks), extracts all text, and embeds it into a vector DB.

    Args:
        datafile_path (str):    path to chunk metadata JSON (e.g., data/processed/metadata/xxx.json) 
        vector_path (str):  Path to FAISS file (e.g., data/processed/vectors/xxx.FAISS)

    Returns:
        vector_path (str): path to vector db faiss file in directory
        index (fiass.Indexflat2): embedded vector
    """

    #read in all chunks for json
    with open(datafile_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    #batches text chunks and embeddeds , saving in a fiass file
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)


    dimension = vectors.shape[1] #dimentions of vector
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    
    os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    faiss.write_index(index, vector_path) #writes embeddings into file

    #logs sucess
    print(f"[LOG] FAISS vector DB saved at: {vector_path}")
    return index, vector_path

def append_embeddings(data: list[dict], vector_path: str) -> str:
    """
    Takes new metadata chunks, embeds them, and appends to an existing FAISS index.

    Args:
        data (list[dict]): appended chunk metadata
        vector_path (str): path to FAISS index file

    Returns:
        index (fiass.Indexflat2): embedded vector
        vector_path (str): path to FAISS index file
    """

    # Extract text and embed
    texts = [chunk["text"] for chunk in data]
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    vectors = np.array(embeddings).astype("float32")
    faiss.normalize_L2(vectors)

    # Load existing FAISS index if it exists, otherwise create new
    dimension = vectors.shape[1]

    if os.path.exists(vector_path):
        index = faiss.read_index(vector_path)
        print("[LOG] Loaded existing FAISS index")
    else:
        index = faiss.IndexFlatL2(dimension)
        print("[LOG] Created new FAISS index")

    # Add new vectors
    index.add(vectors)

    # Ensure directory exists and save index
    os.makedirs(os.path.dirname(vector_path), exist_ok=True)
    faiss.write_index(index, vector_path)
    print(f"[LOG] FAISS vector DB saved at: {vector_path}")

    return index, vector_path