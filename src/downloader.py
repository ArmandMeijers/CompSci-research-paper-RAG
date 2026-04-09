'''
Author: Armand Meijers
Date: 02/04/2026
Description: Downloads a set of Comp Sci Research Papers from arxiv (https://arxiv.org) and stores in folder data/raw/papers
'''

#imports
import arxiv, json, os, time

METADATA_PATH = "data/processed/metadata/"
METADATA_JSON_PATH = "data/processed/metadata/doc_metadata.json"

def download_papers_arxiv(max_papers: int, categories: list, folder_path: str):

    metadata = []
    current_paper = 1 #file index identifier

    #Dowloads n number of docs for each categorie states in the list
    for category in categories:
        query = f"cat:{category}"

        #searches arxiv database for n number of docs related to categirie
        search = arxiv.Search(
            query=query, 
            max_results=max_papers,
            #https://arxiv.org
        )

        #for each result download document
        for paper in search.results():
            filename = f"paper{current_paper}-{category}.pdf"
            filepath = os.path.join(folder_path, filename)

            #checks if there are duplicate files
            if not os.path.exists(filepath):
                #error checking and logging.
                try:
                    paper.download_pdf(dirpath=folder_path, filename=filename)

                    metadata.append({
                        "filename": filename,
                        "title": paper.title,
                        "authors": [a.name for a in paper.authors],
                        "abstract": paper.summary,
                        "arxiv_id": paper.entry_id
                    })

                    print(f"[LOG] Downloaded {filename}")

                except Exception as e:
                    print(f"[ERROR] Failed to download {filename}: {e}")
            
            current_paper += 1 
            time.sleep(1) #preventing excessive requests


    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        print("[LOG] Saved metadata")