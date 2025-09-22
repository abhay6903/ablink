import mysql.connector
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rapidfuzz import fuzz
import gradio as gr


# ------------------------- MySQL Connection -------------------------

def connect_to_mysql():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='abhay123', 
        database='passionbytes'   
    )


# ---------------------- Fetch Data from MySQL ----------------------

def fetch_data():
    conn = connect_to_mysql()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT * FROM entity_records
    """)
    rows = cursor.fetchall()

    df = pd.DataFrame(rows)
    conn.close()
    return df


# ---------------------- Vectorization Logic ------------------------

def vectorize_text_fields(df):
    combined_text = df[['full_name', 'city', 'state', 'address_line1']].fillna('').agg(' '.join, axis=1)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_text)

    return tfidf_matrix


# --------------------- Cosine Similarity+ANN-------------------

def custom_ann_clustering(tfidf_matrix, ids, similarity_threshold=0.5, min_samples=1):
    cosine_sim = cosine_similarity(tfidf_matrix)
    n = len(ids)
    visited = [False] * n
    clusters = {}
    cluster_id = 0
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
    idx_to_id = {idx: id_ for idx, id_ in enumerate(ids)}

    for i in range(n):
        if not visited[i]:
            clusters[cluster_id] = [idx_to_id[i]]
            visited[i] = True
            neighbors = [j for j in range(n) if (not visited[j]) and (cosine_sim[i, j] >= similarity_threshold)]
            for j in neighbors:
                clusters[cluster_id].append(idx_to_id[j])
                visited[j] = True
            cluster_id += 1
    id_to_cluster = {}
    for cid, id_list in clusters.items():
        for rid in id_list:
            id_to_cluster[rid] = cid
    return id_to_cluster


def get_combined_text(row):
    return f"{row['full_name']} {row['city']} {row['state']} {row['address_line1']}"
def fuzzy_refine_clusters(df, initial_clusters, move_threshold=85, merge_threshold=80):
    clusters = {}
    for record_id, cluster in initial_clusters.items():
        clusters.setdefault(cluster, []).append(record_id)

    id_to_text = {row['id']: get_combined_text(row) for _, row in df.iterrows()}

    moved = True
    while moved:
        moved = False
        for record_id, current_cluster in list(initial_clusters.items()):
            text = id_to_text[record_id]
            best_cluster = current_cluster
            best_score = -1
            for cluster, ids in clusters.items():
                if cluster == current_cluster:
                    continue
                scores = [fuzz.token_set_ratio(text, id_to_text[other_id]) for other_id in ids]
                avg_score = np.mean(scores) if scores else 0
                if avg_score > best_score:
                    best_score = avg_score
                    best_cluster = cluster
            if best_score >= move_threshold and best_cluster != current_cluster:
                # Move record
                clusters[current_cluster].remove(record_id)
                clusters.setdefault(best_cluster, []).append(record_id)
                initial_clusters[record_id] = best_cluster
                moved = True
        clusters = {k: v for k, v in clusters.items() if v}

    cluster_ids = list(clusters.keys())
    merged = True
    while merged:
        merged = False
        for i in range(len(cluster_ids)):
            for j in range(i+1, len(cluster_ids)):
                c1, c2 = cluster_ids[i], cluster_ids[j]
                ids1, ids2 = clusters[c1], clusters[c2]
                # Compute average similarity between all pairs
                scores = [fuzz.token_set_ratio(id_to_text[id1], id_to_text[id2]) for id1 in ids1 for id2 in ids2]
                avg_score = np.mean(scores) if scores else 0
                if avg_score >= merge_threshold:
                    clusters[c1].extend(ids2)
                    for rid in ids2:
                        initial_clusters[rid] = c1
                    del clusters[c2]
                    cluster_ids.remove(c2)
                    merged = True
                    break
            if merged:
                break

    
    final_clusters = {}
    cluster_map = {old: new for new, old in enumerate(sorted(clusters.keys()))}
    for record_id, cluster in initial_clusters.items():
        final_clusters[record_id] = cluster_map[cluster]
    return final_clusters


def get_user_record():
    print("\nEnter a structured record (comma-separated)")
    input_str = input("Record: ")
    parts = [p.strip() for p in input_str.split(",")]
    # Map to fields based on expected order
    user_record = {
        'full_name': parts[0] if len(parts) > 0 else '',
        'date_of_birth': parts[1] if len(parts) > 1 else '',
        'email': parts[2] if len(parts) > 2 else '',
        'address_line1': parts[3] if len(parts) > 3 else '',
        'address_line2': parts[4] if len(parts) > 4 else '',
        'address_line3': parts[5] if len(parts) > 5 else '',
        'address_line4': parts[6] if len(parts) > 6 else '',
        'city': parts[7] if len(parts) > 7 else '',
        'state': parts[8] if len(parts) > 8 else '',
        'country': parts[9] if len(parts) > 9 else '',
    }
    # Combine address lines for searching
    user_record['address_full'] = ' '.join([
        user_record.get('address_line1',''),
        user_record.get('address_line2',''),
        user_record.get('address_line3',''),
        user_record.get('address_line4','')
    ]).strip()
    return user_record


# ----------------------------- Main -------------------------------

class DoublyLinkedListNode:
    def __init__(self, idx, vec, record_id):
        self.idx = idx
        self.vec = vec
        self.record_id = record_id
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def append(self, idx, vec, record_id):
        node = DoublyLinkedListNode(idx, vec, record_id)
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            node.prev = self.tail
            self.tail = node
        self.size += 1

    def find_most_similar(self, user_vec):
        max_sim = -1
        best_record_id = None
        node = self.head
        while node:
            sim = np.dot(user_vec, node.vec) / (np.linalg.norm(user_vec) * np.linalg.norm(node.vec) + 1e-10)
            if sim > max_sim:
                max_sim = sim
                best_record_id = node.record_id
            node = node.next
        return best_record_id, max_sim


def search_record(full_name, date_of_birth, email, address_line1, address_line2, address_line3, address_line4, city, state, country):
    df = fetch_data()
    if df.empty:
        return "No records found in database.", "-"
    tfidf_matrix = vectorize_text_fields(df)
    cluster_labels = custom_ann_clustering(tfidf_matrix, df['id'].tolist(), similarity_threshold=0.5)
    refined_labels = fuzzy_refine_clusters(df, cluster_labels)
    all_texts = df[['full_name', 'city', 'state', 'address_line1']].fillna('').agg(' '.join, axis=1).tolist()
    all_ids = df['id'].tolist()
    user_record = {
        'full_name': full_name,
        'date_of_birth': date_of_birth,
        'email': email,
        'address_line1': address_line1,
        'address_line2': address_line2,
        'address_line3': address_line3,
        'address_line4': address_line4,
        'city': city,
        'state': state,
        'country': country,
    }
    user_record['address_full'] = ' '.join([
        user_record.get('address_line1',''),
        user_record.get('address_line2',''),
        user_record.get('address_line3',''),
        user_record.get('address_line4','')
    ]).strip()
    user_text = f"{user_record['full_name']} {user_record['city']} {user_record['state']} {user_record['address_full']}"
    vectorizer = TfidfVectorizer()
    tfidf_matrix_all = vectorizer.fit_transform(all_texts + [user_text])
    user_vec = tfidf_matrix_all[-1].toarray()[0]
    tfidf_matrix_all = tfidf_matrix_all[:-1]
    dll = DoublyLinkedList()
    for i, vec in enumerate(tfidf_matrix_all.toarray()):
        dll.append(i, vec, all_ids[i])
    matched_id, max_sim = dll.find_most_similar(user_vec)
    matched_cluster = refined_labels[matched_id]
    canonical_record = df[df['id'] == matched_id]
    return canonical_record.to_string(index=False), str(matched_cluster)


def search_record_singlebox(input_str):
    parts = [p.strip() for p in input_str.split(",")]
    user_record = {
        'full_name': parts[0] if len(parts) > 0 else '',
        'date_of_birth': parts[1] if len(parts) > 1 else '',
        'email': parts[2] if len(parts) > 2 else '',
        'address_line1': parts[3] if len(parts) > 3 else '',
        'address_line2': parts[4] if len(parts) > 4 else '',
        'address_line3': parts[5] if len(parts) > 5 else '',
        'address_line4': parts[6] if len(parts) > 6 else '',
        'city': parts[7] if len(parts) > 7 else '',
        'state': parts[8] if len(parts) > 8 else '',
        'country': parts[9] if len(parts) > 9 else '',
    }
    user_record['address_full'] = ' '.join([
        user_record.get('address_line1',''),
        user_record.get('address_line2',''),
        user_record.get('address_line3',''),
        user_record.get('address_line4','')
    ]).strip()
    df = fetch_data()
    if df.empty:
        return "No matching record found.", "-"
    tfidf_matrix = vectorize_text_fields(df)
    cluster_labels = custom_ann_clustering(tfidf_matrix, df['id'].tolist(), similarity_threshold=0.5)
    refined_labels = fuzzy_refine_clusters(df, cluster_labels)
    all_texts = df[['full_name', 'city', 'state', 'address_line1']].fillna('').agg(' '.join, axis=1).tolist()
    all_ids = df['id'].tolist()
    user_text = f"{user_record['full_name']} {user_record['city']} {user_record['state']} {user_record['address_full']}"
    vectorizer = TfidfVectorizer()
    tfidf_matrix_all = vectorizer.fit_transform(all_texts + [user_text])
    user_vec = tfidf_matrix_all[-1].toarray()[0]
    tfidf_matrix_all = tfidf_matrix_all[:-1]
    dll = DoublyLinkedList()
    for i, vec in enumerate(tfidf_matrix_all.toarray()):
        dll.append(i, vec, all_ids[i])
    matched_id, max_sim = dll.find_most_similar(user_vec)
    matched_cluster = refined_labels[matched_id]
    canonical_record = df[df['id'] == matched_id]
    if canonical_record.shape[0] > 0:
        record = canonical_record.iloc[0]
        record_str = 'Original record present\n' + '\n'.join([f"{col}: {record[col]}" for col in canonical_record.columns])
        return record_str, str(matched_cluster)
    else:
        return "No matching record found.", "-"


def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# PassionBytes Record Search")
        gr.Markdown("Enter all fields as comma-separated values in the order: Full Name, Date of Birth, Email, Address Line 1, Address Line 2, Address Line 3, Address Line 4, City, State, Country")
        input_box = gr.Textbox(label="Enter record (comma-separated)")
        btn = gr.Button("Search")
        output_record = gr.Textbox(label="Original Record")
        output_cluster = gr.Textbox(label="Cluster ID")
        btn.click(
            search_record_singlebox,
            inputs=input_box,
            outputs=[output_record, output_cluster]
        )
    demo.launch()

if __name__ == '__main__':
    launch_gradio()
