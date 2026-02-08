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
    """Route pour le chatbot RAG"""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        
        if not question:
            return render_template("chat.html", error="Veuillez poser une question.")
        
        try:
            print(f"🔍 Question reçue: '{question}'")
            
            # 1. Recherche sémantique des chunks les plus pertinents
            top_chunks = recherche_semantique(question, top_k=3)
            
            if not top_chunks:
                return render_template("chat.html", 
                                     error="Aucun chunk trouvé. Assurez-vous d'avoir lancé le pipeline /run d'abord.")
            
            print(f"✅ Trouvé {len(top_chunks)} chunks pertinents")
            
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

if __name__ == "__main__":
    app.run(debug=True, port=5000)
