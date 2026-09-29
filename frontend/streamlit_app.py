import requests
import streamlit as st


import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)

st.set_page_config(
    page_title="ShadowFox RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


st.title("🤖 ShadowFox RAG Assistant")
st.caption("Ask questions about your uploaded documents using AI-powered retrieval.")


# --------------------------------------------------
# Upload Document
# --------------------------------------------------

st.header("📄 Upload Document")

uploaded_file = st.file_uploader(
    "Upload a PDF, TXT, or Markdown file",
    type=["pdf", "txt", "md"]
)


if uploaded_file is not None:

    if st.button("📤 Upload and Index Document"):

        with st.spinner("Uploading and indexing document..."):

            try:
                response = requests.post(
                    f"{API_URL}/api/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type
                        )
                    },
                    timeout=300
                )

                if response.status_code == 200:

                    data = response.json()

                    st.success(
                        "Document uploaded and indexed successfully!"
                    )

                    st.session_state["document_name"] = data["filename"]

                    st.write(
                        f"**Chunks created:** {data['chunks_created']}"
                    )

                    st.write(
                        f"**Embedding dimension:** "
                        f"{data['embedding_dimension']}"
                    )

                else:

                    st.error(
                        f"Upload failed: {response.text}"
                    )

            except requests.exceptions.RequestException as error:

                st.error(
                    f"Could not connect to the FastAPI server: {error}"
                )


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

st.header("💬 Ask a Question")

document_name = st.session_state.get(
    "document_name",
    ""
)

if document_name:

    st.info(
        f"Selected document: **{document_name}**"
    )

    question = st.text_input(
        "Enter your question"
    )

    top_k = st.slider(
        "Number of sources to retrieve",
        min_value=1,
        max_value=10,
        value=3
    )

    if st.button("🔍 Ask Question"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching document and generating answer..."
            ):

                try:

                    response = requests.post(
                        f"{API_URL}/api/query",
                        json={
                            "document_name": document_name,
                            "question": question,
                            "top_k": top_k
                        },
                        timeout=300
                    )

                    if response.status_code == 200:

                        result = response.json()

                        st.subheader("🧠 Answer")

                        st.write(
                            result["answer"]
                        )

                        st.subheader(
                            "📚 Retrieved Sources"
                        )

                        for source in result["sources"]:

                            with st.expander(
                                f"Source {source['chunk_id']} "
                                f"— Distance: "
                                f"{source['distance']:.4f}"
                            ):

                                st.write(
                                    f"**Document:** "
                                    f"{source['source']}"
                                )

                                st.write(
                                    source["content"]
                                )

                    else:

                        st.error(
                            f"Query failed: {response.text}"
                        )

                except requests.exceptions.RequestException as error:

                    st.error(
                        f"Could not connect to the FastAPI server: {error}"
                    )

else:

    st.info(
        "👆 Upload and index a document first."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "ShadowFox Advanced AI Engineer Task 3 • "
    "FastAPI + Streamlit + LangGraph + FAISS + Gemini"
)