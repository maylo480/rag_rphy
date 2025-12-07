from key_functions import generate_embeddings_and_upload_to_db, search_by_query
from postgres_connector import get_psql_session
from ollama import chat

def provide_context_for_query(query, session):
    context = search_by_query(
        session=session,
        query=query
    )
    print(f"Context: {context}")

    prompt = f"<|content_start>{context} \
    <|content_end> {query}"

    print(f"Prompt: {prompt}")
    return prompt

def main():
    query = "Tell me anything you know about device called RPD"
    psql_session = get_psql_session(db_name="test_embeddings")
    generate_embeddings_and_upload_to_db("rphy_pdfs", psql_session)
    prompt = provide_context_for_query(query=query, session=psql_session)
    response = chat(model='Llama-3.1-8b-local:latest', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ]
    )

    print(response.message.content)

if __name__ == "__main__":
    main()