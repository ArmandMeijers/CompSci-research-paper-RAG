'''
Author: Armand Meijers
Date: 03/04/2026
Description: main file that puts all fucntions toegther to run full pipeline
'''

from src import ingestion, retriever, downloader
from src import helper
import json, os, faiss

DOCUMENT_PATH = "data/raw/documents/"
METADATA_PATH = "data/processed/metadata/"
VECTOR_PATH = "data/processed/vectors/"
INDEX_PATH = "data/processed/vectors/index.faiss"
CHUNK_JSON_PATH = "data/processed/metadata/chunk_metadata.json"
DOC_JSON_PATH = "data/processed/metadata/doc_metadata.json"

MAX_PAPERS = 30
CATEGORIES = ["cs.LG","cs.AI","cs.CL","cs.CV"] #types of papers searched for.
#"cs.LG","cs.AI","cs.CL","cs.CV"

#function checks if ingestion exists and if more docs have been added to foler, updates vector db
def load_up():
    #ensures paths exist and if not creares them
    paths = [DOCUMENT_PATH, METADATA_PATH, VECTOR_PATH, INDEX_PATH, CHUNK_JSON_PATH, DOC_JSON_PATH] #list of folder/file paths
    for path in paths:
        helper.path_checker_creator(path)
    
    #Creates array of all pdfs in folder at path
    pdfs = [f for f in os.listdir(DOCUMENT_PATH) if f.endswith(".pdf")]
    #if no PDFs files ask use if they want to download files / add your own
    if not pdfs:
        user_choice = str(input("No files found. Download example papers? (y/n): ").lower())
        if user_choice in ["y", "yes"]:
            downloader.download_papers_arxiv(MAX_PAPERS, CATEGORIES, DOCUMENT_PATH)            
        else:
            print("Add files to data/raw/papers and rerun")
            return False
        
    #checks if chunking file exist and is empty
    if os.path.exists(CHUNK_JSON_PATH) and os.path.getsize(CHUNK_JSON_PATH) == 0:
        #run chunking on all pdfs in folder
        chunk_metadata = ingestion.chunking_files_pdf(DOCUMENT_PATH)
        if chunk_metadata == 0: # error check
            return False
      
        #runs embedding modle on chunks and saves it so a faiss file
        index, embedding_path = ingestion.embedding_text(CHUNK_JSON_PATH, INDEX_PATH)

    #if file exists but is not empty run check that all processed data is up-to-date
    else:
        #gets current metadata
        with open(DOC_JSON_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        #saves filename param to compare to current pdfs
        existing_files = {m["filename"] for m in metadata}

        #checks if current files all match previously saved names
        newPDFs = [
            os.path.join(DOCUMENT_PATH, f)
            for f in pdfs
            if f not in existing_files
        ]

        #checks if new pdfs or not
        if not newPDFs:
            print("[LOG] Embedding up to date!")
        else:
            #runs a append vesion of chunking and embedding to add new processed data
            new_data = ingestion.append_chunks_pdf(newPDFs, CHUNK_JSON_PATH) #chunks new files and appends metadata
            ingestion.append_embeddings(new_data, INDEX_PATH)
    
def main():
    load_up()
    index = ingestion.load_or_create_index(INDEX_PATH, CHUNK_JSON_PATH)
    if index == False:
        return False


    query, user_vector = retriever.prompt_user_query()
    distances, indices = retriever.cosine_similarity(index, user_vector)

    with open(CHUNK_JSON_PATH, "r", encoding="utf-8") as f:
        chunk_metadata = json.load(f)

    for idx, dist in zip(indices[0], distances[0]):
        chunk = chunk_metadata[idx]["text"]
        print(f"Distance: {dist:.3f} | Text: {chunk}...")

    
if __name__ == "__main__":
    main()