from dotenv import load_dotenv
import os
import json
from flask import Flask, render_template, request, url_for
from main import main
from stockage import recuperer_tous_les_vecteurs, recherche_semantique
from visualisation import visualisation_2d_to_file
from embeddings import get_embeddings
from groq_chat import generer_reponse_groq
import stockage

app = Flask(__name__)

dotenv_path = "../.env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_pipeline():
    chunks = main()
    return render_template("results.html", nb_chunks=len(chunks))

@app.route("/visu-vector")
def visu_vector():
    static_dir = os.path.join(app.static_folder, "visu")
    os.makedirs(static_dir, exist_ok=True)
    image_path = os.path.join(static_dir, "vector.png")
    error = None
    try:
        chunks = recuperer_tous_les_vecteurs()
        if not chunks:
            error = "Aucun vecteur. Lance /run d'abord."
        else:
            visualisation_2d_to_file(chunks, image_path)
    except Exception as e:
        error = str(e)
    return render_template("visu_vector.html",
                         image_url=url_for("static", filename="visu/vector.png") if not error else None,
                         error=error)

@app.route("/chat", methods=["GET", "POST"])
def chat():
    """Route pour le chatbot RAG avec recherche par clusters"""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        
        if not question:
            return render_template("chat.html", error="Veuillez poser une question.")
        
        try:
            print(f"🔍 Question reçue: '{question}'")
            
            # 1. Recherche par clusters des chunks les plus pertinents
            top_chunks = recherche_par_clusters(
                question, 
                top_clusters=2,      # Sélectionner les 2 meilleurs clusters
                chunks_per_cluster=2  # Prendre 2 chunks par cluster
            )
            
            if not top_chunks:
                return render_template("chat.html", 
                                     error="Aucun chunk trouvé. Assurez-vous d'avoir lancé le pipeline /run d'abord.")
            
            print(f"✅ Trouvé {len(top_chunks)} chunks pertinents via clustering")
            
            # 2. Générer la réponse avec Groq
            reponse = generer_reponse_groq(question, top_chunks)
            
            # 3. Extraire les similarités pour l'affichage
            similarites = [f"{chunk.get('certainty', 0):.2%}" for chunk in top_chunks]
            
            print(f"✅ Réponse générée avec succès")
            
            return render_template("chat.html",
                                 question=question,
                                 top_chunks=top_chunks,
                                 similarites=similarites,
                                 reponse=reponse)
            
        except Exception as e:
            print(f"❌ Erreur dans le chatbot: {e}")
            import traceback
            traceback.print_exc()
            return render_template("chat.html", 
                                 error=f"Erreur lors du traitement: {str(e)}")
    
    # GET request - afficher le formulaire vide
    return render_template("chat.html")

import numpy as np
from flask import render_template, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from clustering import clustering
from stockage import recuperer_tous_les_vecteurs
from main import generate_cluster_names


def recherche_par_clusters(question, top_clusters=2, chunks_per_cluster=2):
    try:
        all_chunks = recuperer_tous_les_vecteurs()
        if not all_chunks:
            return []
        
        clusters = clustering(all_chunks)
        print(f"🔍 {len(clusters)} clusters trouvés")
        
        # 2. Calculer l'embedding de la question
        question_vector = get_embeddings(question)
        if hasattr(question_vector, 'tolist'):
            question_vector = question_vector.tolist()
        if isinstance(question_vector, list) and len(question_vector) > 0 and isinstance(question_vector[0], (list, tuple)):
            question_vector = question_vector[0]
        question_vector = np.array(question_vector)
        
        # 3. Calculer le centre de chaque cluster et la similarité avec la question
        cluster_similarities = []
        for cluster_id, chunks in clusters.items():
            # Extraire les vecteurs du cluster
            vectors = []
            for chunk in chunks:
                vector = chunk.get("vector")
                if isinstance(vector, dict):
                    vector = list(vector.values())[0]
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                vectors.append(vector)
            
            # Calculer le centre du cluster
            cluster_center = np.mean(vectors, axis=0)
            
            # Calculer la similarité cosinus entre la question et le centre du cluster
            similarity = cosine_similarity(
                question_vector.reshape(1, -1),
                cluster_center.reshape(1, -1)
            )[0][0]
            
            cluster_similarities.append({
                'cluster_id': cluster_id,
                'similarity': similarity,
                'chunks': chunks
            })
        
        # 4. Trier les clusters par similarité décroissante
        cluster_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_relevant_clusters = cluster_similarities[:top_clusters]
        
        print(f"✅ Top {top_clusters} clusters sélectionnés")
        for i, cluster_info in enumerate(top_relevant_clusters):
            print(f"  Cluster {cluster_info['cluster_id']}: similarité {cluster_info['similarity']:.2%}")
        
        # 5. Dans chaque cluster pertinent, trouver les chunks les plus pertinents
        selected_chunks = []
        for cluster_info in top_relevant_clusters:
            chunks = cluster_info['chunks']
            
            # Calculer la similarité de chaque chunk avec la question
            chunk_similarities = []
            for chunk in chunks:
                vector = chunk.get("vector")
                if isinstance(vector, dict):
                    vector = list(vector.values())[0]
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                chunk_vector = np.array(vector)
                
                similarity = cosine_similarity(
                    question_vector.reshape(1, -1),
                    chunk_vector.reshape(1, -1)
                )[0][0]
                
                chunk_similarities.append({
                    'text': chunk.get('text'),
                    'metadata': chunk.get('metadata'),
                    'certainty': similarity,
                    'cluster_id': cluster_info['cluster_id']
                })
            
            # Trier et prendre les meilleurs chunks de ce cluster
            chunk_similarities.sort(key=lambda x: x['certainty'], reverse=True)
            selected_chunks.extend(chunk_similarities[:chunks_per_cluster])
        
        # 6. Trier tous les chunks sélectionnés par similarité
        selected_chunks.sort(key=lambda x: x['certainty'], reverse=True)
        
        print(f"✅ {len(selected_chunks)} chunks sélectionnés au total")
        
        return selected_chunks
        
    except Exception as e:
        print(f"❌ Erreur dans recherche_par_clusters: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route("/bulle", methods=["GET", "POST"])
def bulle():
    """
    Route pour afficher la visualisation des clusters sous forme de bulles
    """
    return render_template("bulle.html")


@app.route("/api/clusters-data", methods=["GET"])
def get_clusters_data():
    """
    API pour récupérer les données des clusters formatées pour la visualisation
    """
    try:
        # Récupérer les clusters depuis ta base de données ou session
        # Adapter selon ton implémentation
        new_chunk = recuperer_tous_les_vecteurs()
        final_clusters = clustering(new_chunk)
        print("Cluster récupéré")
        cluster_names = generate_cluster_names(final_clusters)  # À implémenter selon ton code
        
        if not final_clusters or not cluster_names:
            return jsonify({
                "error": "Aucun cluster disponible. Lancez d'abord le clustering."
            }), 404
        
        # Calculer les centres des clusters
        cluster_centers = {}
        cluster_vectors = {}
        
        for cluster_id, chunks in final_clusters.items():
            vectors = []
            for chunk in chunks:
                vector = chunk.get("vector")
                if isinstance(vector, dict):
                    vector = list(vector.values())[0]
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                vectors.append(vector)
            
            if vectors:
                cluster_vectors[cluster_id] = np.array(vectors)
                cluster_centers[cluster_id] = np.mean(cluster_vectors[cluster_id], axis=0)
        
        # Normaliser les centres pour cosine similarity
        center_ids = list(cluster_centers.keys())
        center_matrix = np.array([cluster_centers[cid] for cid in center_ids])
        center_matrix_norm = normalize(center_matrix)
        
        # Calculer la similarité entre tous les clusters
        similarity_matrix = cosine_similarity(center_matrix_norm)
        
        # Créer les nœuds (bulles)
        nodes = []
        for i, cluster_id in enumerate(center_ids):
            cluster_size = len(final_clusters[cluster_id])
            cluster_name = cluster_names[i] if i < len(cluster_names) else f"Cluster {cluster_id}"
            
            # Extraire quelques exemples de textes
            sample_texts = [
                chunk.get("text", "")[:100] + "..." 
                for chunk in final_clusters[cluster_id][:3]
            ]
            
            nodes.append({
                "id": str(cluster_id),
                "name": cluster_name,
                "size": cluster_size,
                "samples": sample_texts,
                "chunk_count": cluster_size
            })
        
        # Créer les liens (edges) entre clusters proches
        links = []
        threshold = 0.4  # Seuil de similarité (ajustable)
        
        for i in range(len(center_ids)):
            # Trouver les 3 clusters les plus similaires
            similarities = []
            for j in range(len(center_ids)):
                if i != j:
                    similarities.append({
                        'index': j,
                        'similarity': similarity_matrix[i][j]
                    })
            
            # Trier par similarité décroissante
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Garder les 3 plus proches au-dessus du seuil
            for sim in similarities[:3]:
                if sim['similarity'] > threshold:
                    # Éviter les doublons (i->j et j->i)
                    link_id = tuple(sorted([center_ids[i], center_ids[sim['index']]]))
                    
                    if not any(tuple(sorted([l['source'], l['target']])) == link_id for l in links):
                        links.append({
                            "source": str(center_ids[i]),
                            "target": str(center_ids[sim['index']]),
                            "strength": float(sim['similarity'])
                        })
        
        return jsonify({
            "nodes": nodes,
            "links": links,
            "total_clusters": len(nodes),
            "total_links": len(links)
        })
    
    except Exception as e:
        print(f"Erreur dans get_clusters_data: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
