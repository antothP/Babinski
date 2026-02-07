import os
from flask import Flask, render_template, request, url_for
from main import main
from stockage import recuperer_tous_les_vecteurs
from visualisation import visualisation_2d_to_file

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run_pipeline():
    chunks = main()
    return render_template(
        "results.html",
        nb_chunks=len(chunks),
    )


@app.route("/visu-vector")
def visu_vector():
    """Affiche la page de visualisation vectorielle (PCA des embeddings)."""
    static_dir = os.path.join(app.static_folder, "visu")
    os.makedirs(static_dir, exist_ok=True)
    image_path = os.path.join(static_dir, "vector.png")
    error = None
    try:
        chunks = recuperer_tous_les_vecteurs()
        if not chunks:
            error = "Aucun vecteur en base. Lancez d'abord le pipeline pour ingérer des documents."
        else:
            visualisation_2d_to_file(chunks, image_path)
    except Exception as e:
        error = str(e)
    return render_template(
        "visu_vector.html",
        image_url=url_for("static", filename="visu/vector.png") if not error else None,
        error=error,
    )

if __name__ == "__main__":
    app.run(debug=True)