from postgres_connector import TextEmbedding

def get_surrounding_sentences(entry_ids, file_names, group_window_size, session):
    surrounding_sentences = []
    for entry_id, file_name in zip(entry_ids, file_names):
        surrounding_sentences.append(
            session.query(TextEmbedding.id, TextEmbedding.sentence_number, TextEmbedding.content, TextEmbedding.file_name)\
            .filter(TextEmbedding.id >= entry_id - group_window_size)\
            .filter(TextEmbedding.id <= entry_id + group_window_size)\
            .filter(TextEmbedding.file_name == file_name).all()
        )
    return surrounding_sentences

# Finding the content from our database which is most similar to the query
def search_embeddings(query_embedding, session, limit=5):
    return session.query(TextEmbedding.id, TextEmbedding.sentence_number, TextEmbedding.content, TextEmbedding.file_name, 
        TextEmbedding.embedding.cosine_distance(query_embedding).label("distance") )\
        .order_by("distance").limit(limit).all()

# Check if a matches context window overlaps with another matches context window.
def is_unique_to_window(existing_matches, current_match, group_window_size=5):
    for match in existing_matches:
        if match[3] != current_match[3]:
            continue
        if match[1] > current_match[1] + group_window_size or match[1] < current_match[1] - group_window_size:
            continue
        else:
            return False
    return True

# Getting unique matches from search results
def get_filtered_matches(search_results):
    unique_count = 0
    matches = []
    for result in search_results:
        if unique_count >= 5:
            break
        if is_unique_to_window(matches, result):
            unique_count += 1
        matches.append(result)
    return matches