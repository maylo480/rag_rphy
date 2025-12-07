from sentence_transformers import SentenceTransformer
import nltk
import os
import gc
from pdf_extraction import convert_single_pdf_to_sentences
from postgres_connector import upload_embeddings_into_db, search_embeddings
from utils import get_filtered_matches, get_surrounding_sentences

def generate_embeddings_and_upload_to_db(pdfs_path, psql_session) -> None:
    try:
        #print("Initializing model")
        nltk.download("punkt")
        nltk.download("punkt_tab")
        model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    except Exception as e:
        print(f"Error during initializing model: {str(e)}")

    for file in os.listdir(pdfs_path):
        if not file.endswith("pdf"):
            continue
        filename = os.path.join(pdfs_path, file)
        try:
            print(f"Generating embeddings for file {filename}")
            sentences = convert_single_pdf_to_sentences(filename)
            # embeddings = embed(model="custom_deepseek", input=sentences)["embeddings"]
            embeddings = model.encode(sentences, show_progress_bar=True)
            #print(f"Succesfully generated embeddings for: {filename}")
            #print(f"Uploading embedding for file {filename}")
            upload_embeddings_into_db(
                session=psql_session,
                embeddings=embeddings,
                sentences=sentences,
                filename=filename,
            )
            #print(f"Uploaded embeddings for file {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def search_by_query(session, query, num_matches=5, group_window_size=5):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    query_embedding = model.encode(query)
    del model
    gc.collect()

    search_results = search_embeddings(query_embedding, session=session, limit=num_matches * (2*group_window_size + 1) )
    filtered_matches = get_filtered_matches(search_results)

    entry_ids = [i[0] for i in filtered_matches]
    file_names = [i[3] for i in filtered_matches]

    return get_surrounding_sentences(entry_ids=entry_ids, file_names=file_names, group_window_size=group_window_size, session=session)