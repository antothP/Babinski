#!/usr/bin/env python3
"""
Script de test pour vérifier que tous les composants sont correctement installés
"""

import os
import sys

def test_imports():
    """Teste que tous les modules peuvent être importés"""
    print("🔍 Test des imports...")
    
    modules = [
        ('flask', 'Flask'),
        ('dotenv', 'python-dotenv'),
        ('groq', 'Groq'),
        ('weaviate', 'Weaviate'),
        ('ollama', 'Ollama'),
    ]
    
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - Installation requise: pip install {module}")
            return False
    
    return True

def test_env_vars():
    """Teste que les variables d'environnement sont définies"""
    print("\n🔍 Test des variables d'environnement...")
    
    from dotenv import load_dotenv
    load_dotenv("../.env")
    
    if os.getenv("GROQ_API_KEY"):
        print("  ✅ GROQ_API_KEY définie")
        return True
    else:
        print("  ❌ GROQ_API_KEY manquante dans .env")
        return False

def test_weaviate():
    """Teste la connexion à Weaviate"""
    print("\n🔍 Test de la connexion Weaviate...")
    
    try:
        import weaviate
        client = weaviate.connect_to_local(host="localhost", port=8080)
        
        if client.is_ready():
            print("  ✅ Weaviate connecté")
            collections = client.collections.list_all()
            print(f"  📊 Collections disponibles: {list(collections.keys())}")
            client.close()
            return True
        else:
            print("  ❌ Weaviate n'est pas prêt")
            return False
    except Exception as e:
        print(f"  ❌ Erreur de connexion Weaviate: {e}")
        print("  💡 Assurez-vous que Weaviate tourne sur localhost:8080")
        return False

def test_ollama():
    """Teste qu'Ollama est disponible"""
    print("\n🔍 Test Ollama...")
    try:
        import ollama
        # Liste des modèles disponibles
        models = ollama.list()
        model_names = [m['name'] for m in models.get('models', [])]
        
        if 'embeddinggemma' in str(model_names):
            print("  ✅ Modèle embeddinggemma disponible")
            return True
        else:
            print("  ❌ Modèle embeddinggemma non trouvé")
            print("  💡 Installez-le avec: ollama pull embeddinggemma")
            return False
    except Exception as e:
        print(f"  ❌ Erreur Ollama: {e}")
        print("  💡 Assurez-vous qu'Ollama est installé et en cours d'exécution")
        return False

def test_groq():
    """Teste la connexion à l'API Groq"""
    print("\n🔍 Test de l'API Groq...")
    
    try:
        from groq import Groq
        from dotenv import load_dotenv
        load_dotenv("../.env")
        
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        
        # Test simple
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama-3.3-70b-versatile",
            max_tokens=10
        )
        
        if response.choices:
            print("  ✅ API Groq fonctionnelle")
            return True
        else:
            print("  ❌ Réponse vide de Groq")
            return False
            
    except Exception as e:
        print(f"  ❌ Erreur API Groq: {e}")
        print("  💡 Vérifiez votre GROQ_API_KEY dans le fichier .env")
        return False

def main():
    """Lance tous les tests"""
    print("="*60)
    print("🧪 TESTS DU CHATBOT RAG")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Variables d'environnement", test_env_vars()))
    results.append(("Weaviate", test_weaviate()))
    results.append(("Ollama", test_ollama()))
    results.append(("Groq", test_groq()))
    
    print("\n" + "="*60)
    print("📊 RÉSUMÉ")
    print("="*60)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 Tous les tests sont passés ! Vous pouvez lancer l'application.")
        print("   Commande: python app.py")
        return 0
    else:
        print("\n⚠️  Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
