from flask import Flask, render_template, request
from main import main

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
        graph_path=graph_path
    )

if __name__ == "__main__":
    app.run(debug=True)
