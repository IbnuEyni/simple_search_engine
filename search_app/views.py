import re
import string

import docx
import nltk
import numpy as np
import pandas as pd
import PyPDF2
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

from .models import Document

# Preprocess and clean data
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("punkt")

stopwords_list = stopwords.words("english")

english_stopset = set(stopwords.words("english")).union(
    {
        "things",
        "that's",
        "something",
        "take",
        "don't",
        "may",
        "want",
        "you're",
        "set",
        "might",
        "says",
        "including",
        "lot",
        "much",
        "said",
        "know",
        "good",
        "step",
        "often",
        "going",
        "thing",
        "things",
        "think",
        "back",
        "actually",
        "better",
        "look",
        "find",
        "right",
        "example",
    }
)


lemmatizer = WordNetLemmatizer()


def preprocess_data(docs):
    documents_clean = []
    for d in docs:
        document_test = re.sub(r"[^\x00-\x7F]+", " ", d)
        document_test = re.sub(r"@\w+", "", document_test)
        document_test = document_test.lower()
        document_test = document_test = re.sub(
            r"[%s]" % re.escape(string.punctuation), " ", document_test
        )
        document_test = re.sub(r"[0-9]", "", document_test)
        document_test = re.sub(r"\s{2,}", " ", document_test)
        documents_clean.append(document_test)

    processed_docs = [
        " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        for text in documents_clean
    ]
    return processed_docs


def upload_document_page(request):
    return render(request, "upload_document.html")


@csrf_exempt
def upload_and_process_documents(request):
    if request.method == "POST":
        if "document" in request.FILES:
            file = request.FILES["document"]
            file_extension = file.name.split(".")[-1].lower()
            print(
                f"[INFO] Uploading document: {file.name} with extension: {file_extension}"
            )

            if file_extension == "pdf":
                text = extract_text_from_pdf(file)
            elif file_extension == "docx":
                text = extract_text_from_docx(file)
            elif file_extension == "txt":
                text = extract_text_from_txt(file)
            else:
                return JsonResponse({"error": "Unsupported file format"}, status=400)

            # Save document to database
            Document.objects.create(
                title=file.name, file_type=file_extension, content=text
            )
            print(f"[INFO] Document saved: {file.name}")

            return redirect("query_page")
        else:
            return JsonResponse({"error": "No file uploaded"}, status=400)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


def query_page(request):
    return render(request, "query_page.html")


@csrf_exempt
def process_query(request):
    if request.method == "POST":
        query = request.POST.get("query", "")
        n = int(request.POST.get("n", 1))
        print(f"[INFO] Processing query: {query}")
        doc_texts = [
            "i loved you ethiopia, stored elements in Compress find Sparse Ethiopia is the greatest country in the world of nation at universe",
            "also, sometimes, the same words can have multiple different 'lemma's. So, based on the context it's used, you should identify the \
        part-of-speech (POS) tag for the word in that specific context and extract the appropriate lemma. Examples of implementing this comes \
        in the following sections countries.ethiopia With a planned.The name that the Blue Nile river loved took in Ethiopia is derived from the \
        Geez word for great to imply its being the river of rivers The word Abay still exists in ethiopia major languages",
            "With more than  million people, ethiopia is the second most populous nation in Africa after Nigeria, and the fastest growing \
         economy in the region. However, it is also one of the poorest, with a per capita income",
            "The primary purpose of the dam ethiopia is electricity production to relieve Ethiopia’s acute energy shortage and for electricity export to neighboring\
         countries.ethiopia With a planned.",
            "The name that the Blue Nile river loved takes in Ethiopia 'abay' is derived from the Geez blue loved word for great to imply its being the river of rivers The \
         word Abay still exists in Ethiopia major languages to refer to anything or anyone considered to be superior.",
            "Two non-upgraded loved turbine-generators with MW each are the first loveto go into operation with loved MW delivered to the national power grid. This early power\
         generation will start well before the completion",
        ]

        titles = [
            "Two upgraded",
            "Loved Turbine-Generators",
            "Operation With Loved",
            "National",
            "Power Grid",
            "Generator",
        ]

        if not doc_texts:
            return JsonResponse({"error": "No documents available"}, status=400)

        new_docs = preprocess_data(doc_texts)
        global new_titles
        new_titles = [
            " ".join([lemmatizer.lemmatize(doc) for doc in text.split(" ")])
            for text in titles
        ]

        # Initialize TfidfVectorizer
        global vectorizer
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=0.002,
            max_df=0.99,
            max_features=1000,
            lowercase=True,
            stop_words=stopwords_list,
        )

        # Fit and transform documents
        X = vectorizer.fit_transform(new_docs)
        df = pd.DataFrame(X.T.toarray())

        # Query processing
        lemma_ops = " ".join(
            [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(query)]
        )
        print(f"[INFO] Lemmatized query: {lemma_ops}")

        result = get_similar_articles(lemma_ops, new_docs, df, new_titles, n)
        print(f"[INFO] Query results: {result}")
        return JsonResponse({"result": result})
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)


def get_similar_articles(query, doc_texts, df, titles, n):
    # Transform the query to the vector space
    q_vec = (
        vectorizer.transform([query])
        .toarray()
        .reshape(
            df.shape[0],
        )
    )
    # Calculate similarity
    sim = {}
    titl = {}

    for i in range(len(doc_texts)):
        sim[i] = np.dot(df.loc[:, i].values, q_vec) / (
            np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        )  # Calculate the similarity
        titl[i] = np.dot(df.loc[:, i].values, q_vec) / (
            np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
        )

    sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)[:n]

    results = []
    for i, v in sim_sorted:
        results.append({"title": titles[i], "content": doc_texts[i], "similarity": v})

    return results


# Extract text functions
def extract_text_from_pdf(file):
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
        print("[INFO] Extracted text from PDF")
        return text
    except Exception as e:
        return f"Error extracting PDF content: {e}"


def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        print("[INFO] Extracted text from DOCX")
        return text
    except Exception as e:
        return f"Error extracting DOCX content: {e}"


def extract_text_from_txt(file):
    try:
        text = file.read().decode("utf-8")
        print("[INFO] Extracted text from TXT")
        return text
    except Exception as e:
        return f"Error extracting TXT content: {e}"
