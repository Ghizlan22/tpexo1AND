from django.shortcuts import render
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from django.shortcuts import render
import matplotlib.pyplot as plt
import io
import urllib, base64
import seaborn as sns
import numpy as np
def generate_graph(data, title, x_label, y_label, cmap="Blues"):
    """Fonction pour générer un graphe"""
    plt.figure(figsize=(8, 5))
    data.plot(kind='bar', legend=False)
    #sns.heatmap(data, annot=True, cmap=cmap, fmt=".0f", cbar=True, square=True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()

    # Sauvegarder le graphe dans un buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Convertir le graphe en une chaîne encodée en base64
    graph = base64.b64encode(image_png).decode('utf-8')
    plt.close()
    return graph


def generate_scatter(data, title, x_label, y_label):
    """Fonction pour générer un nuage de points"""
    plt.figure(figsize=(10, 8))
    for idx, row in data.iterrows():
        plt.scatter(range(1, len(row) + 1), row, label=idx)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(title="Lignes", bbox_to_anchor=(1, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Sauvegarder le nuage de points dans un buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    # Convertir le nuage de points en une chaîne encodée en base64
    graph = base64.b64encode(image_png).decode('utf-8')
    plt.close()
    return graph

def display_tables(request):
    # Table de données
    tableau = [
        ["o", "s"],
        ["n", "j"],
        ["o", "r"],
        ["o", "t"],
        ["n", "j"],
        ["o", "r"],
        ["o", "s"]
    ]
    t = pd.DataFrame(tableau, columns=["rep1", "rep2"], index=["I1", "I2", "I3", "I4", "I5", "I6", "I7"])
    t.index.name = "Index"  # Ajouter un nom pour l'index
    

    # Codage
    codage = {
        "o": [1, 0],
        "n": [0, 1],
        "j": [1, 0, 0, 0],
        "r": [1, 1, 0, 0],
        "s": [1, 1, 1, 0],
        "t": [1, 1, 1, 1]
    }
    table_codage = []
    for _, row in t.iterrows():
        code_rep1 = codage[row['rep1']]
        code_rep2 = codage[row['rep2']]
        ligne_codage = code_rep1 + code_rep2
        table_codage.append(ligne_codage)

    modalites = ["oui", "non", "jamais", "rarement", "souvent", "toujours"]
    df_codage = pd.DataFrame(table_codage, columns=modalites, index=t.index)
    df_codage.index.name = "Index"  # Ajouter un nom pour l'index
    # Générer la matrice des distances
    distance_matrix = pd.DataFrame(0, index=df_codage.index, columns=df_codage.index, dtype=float)
    for i in df_codage.index:
        for j in df_codage.index:
            if i != j:
                diff_count = sum(df_codage.loc[i] != df_codage.loc[j])
                distance = diff_count / 6
                distance_matrix.loc[i, j] = distance

    # Générer la table de codage complet
    codage_complet = []
    moda = ["jamais", "rarement", "souvent", "toujours"]
    for _, row in df_codage.iterrows():
        nouvelle_ligne = [row["oui"], row["non"], 0, 0, 0, 0]
        for i, modalite in reversed(list(enumerate(moda))):
            if row[modalite] == 1:
                nouvelle_ligne[i + 2] = 1
                break
        codage_complet.append(nouvelle_ligne)

    df_codage_complet = pd.DataFrame(codage_complet, columns=modalites, index=df_codage.index)
    # Graphiques basés sur les anciennes données
    graph1 = generate_graph(
        t.apply(pd.Series.value_counts).fillna(0),
        "Distribution des Réponses",
        "Réponses",
        "Fréquence"
    )
    graph2 = generate_graph(
        df_codage.sum(),
        "Somme des Modalités (Table de Codage)",
        "Modalités",
        "Total"
    )
    graph3 = generate_graph(
        distance_matrix.sum(axis=1),
        "Somme des Distances",
        "Index",
        "Distance Totale"
    )
    graph4 = generate_graph(
        df_codage_complet.sum(),
        "Codage Complet (Somme par Modalité)",
        "Modalités",
        "Total"
    )

    # Calcul de la table de Burt
    X = df_codage_complet.values
    X_transposer = X.T
    burt = np.dot(X_transposer, X)
    modalites = ["oui", "non", "jamais", "rarement", "souvent", "toujours"]
    df_burt = pd.DataFrame(burt, index=modalites, columns=modalites)
    # Heatmap pour la table de Burt
    heatmap_burt = generate_graph(df_burt, "Table de Burt - Heatmap", "Modalités", "Modalités", cmap="Blues")
    # Table de contingence 1
    indices_oui_non = [0, 1]  # Indices pour "oui" et "non"
    indices_moda = [2, 3, 4, 5]  # Indices pour "jamais", "rarement", "souvent", "toujours"
    table_contingence_1 = burt[np.ix_(indices_oui_non, indices_moda)]
    df_contingence_1 = pd.DataFrame(table_contingence_1, index=["oui", "non"], columns=["jamais", "rarement", "souvent", "toujours"])

    #Heatmap pour la table de contingence 1
    heatmap_contingence_1 = generate_graph(df_contingence_1, "Table de Contingence 1 - Heatmap", "Modalités", "Oui / Non", cmap="Greens")
    # Table de contingence 2
    table_contingence_2 = burt[np.ix_(indices_moda, indices_oui_non)]
    df_contingence_2 = pd.DataFrame(table_contingence_2, index=["jamais", "rarement", "souvent", "toujours"], columns=["oui", "non"])

    # Heatmap pour la table de contingence 2
    heatmap_contingence_2 = generate_graph(df_contingence_2, "Table de Contingence 2 - Heatmap", "Oui / Non", "Modalités", cmap="Oranges")
    # Table de fréquence 1
    total_contingence_1 = df_contingence_1.values.sum()
    df_fij_1 = df_contingence_1 / total_contingence_1
    df_fij_1["fi."] = df_fij_1.sum(axis=1)
    df_fij_1.loc["f.j"] = df_fij_1.sum(axis=0)

    # Table de fréquence 2
    total_contingence_2 = df_contingence_2.values.sum()
    df_fij_2 = df_contingence_2 / total_contingence_2
    df_fij_2["fi."] = df_fij_2.sum(axis=1)
    df_fij_2.loc["f.j"] = df_fij_2.sum(axis=0)

    # Profils ligne
    df_profile_ligne = df_fij_1.div(df_fij_1["fi."], axis=0).drop("f.j", axis=0)

    # Profils colonne
    df_profile_colonne = df_fij_1.div(df_fij_1.loc["f.j"], axis=1).drop(columns=["fi."])

    # --- Génération des graphiques ---
    # Heatmaps
    heatmap_fij_1 = generate_graph(df_fij_1, "Tableau de Fréquence 1", "Modalités", "Oui / Non", cmap="Blues")
    heatmap_fij_2 = generate_graph(df_fij_2, "Tableau de Fréquence 2", "Oui / Non", "Modalités", cmap="Greens")
    # Nuage de points profils ligne
    plt.figure(figsize=(8, 6))
    for idx, row in df_profile_ligne.iterrows():
        plt.scatter(df_profile_ligne.columns[:-1], row[:-1], label=idx)
    plt.title("Nuage de points - Profils Ligne")
    plt.xlabel("Modalités")
    plt.ylabel("Valeurs Normalisées")
    plt.legend(title="Lignes")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    profile_ligne_graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    # Nuage de points profils colonne
    plt.figure(figsize=(8, 6))
    for col in df_profile_colonne.columns:
        plt.scatter(df_profile_colonne.index, df_profile_colonne[col], label=col)
    plt.title("Nuage de points - Profils Colonne")
    plt.xlabel("Lignes")
    plt.ylabel("Valeurs Normalisées")
    plt.legend(title="Colonnes")
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    profile_colonne_graph = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()






    
    # --- Conversion des tables en HTML ---
    table_html = t.to_html(classes='table table-striped', border=0)
    codage_html = df_codage.to_html(classes='table table-striped', border=0)
    distance_html = distance_matrix.to_html(classes='table table-striped', border=0)
    codage_complet_html = df_codage_complet.to_html(classes='table table-striped', border=0)
    burt_html = df_burt.to_html(classes='table table-striped', border=0)
    contingence_1_html = df_contingence_1.to_html(classes='table table-striped', border=0)
    contingence_2_html = df_contingence_2.to_html(classes='table table-striped', border=0)
    fij_1_html = df_fij_1.to_html(classes='table table-striped', border=0)
    fij_2_html = df_fij_2.to_html(classes='table table-striped', border=0)
    profile_ligne_html = df_profile_ligne.to_html(classes='table table-striped', border=0)
    profile_colonne_html = df_profile_colonne.to_html(classes='table table-striped', border=0)

    return render(request, 'tables/display.html', {
        # Tables principales
        'table_html': table_html,
        'codage_html': codage_html,
        'distance_html': distance_html,
        'codage_complet_html': codage_complet_html,
        'burt_html': burt_html,
        'contingence_1_html': contingence_1_html,
        'contingence_2_html': contingence_2_html,
        'fij_1_html': fij_1_html,
        'fij_2_html': fij_2_html,
        'profile_ligne_html': profile_ligne_html,
        'profile_colonne_html': profile_colonne_html,
        # Graphiques
        'graph1': graph1,
        'graph2': graph2,
        'graph3': graph3,
        'graph4': graph4,
        'heatmap_burt': heatmap_burt,
        'heatmap_contingence_1': heatmap_contingence_1,
        'heatmap_contingence_2': heatmap_contingence_2,
        'heatmap_fij_1': heatmap_fij_1,
        'heatmap_fij_2': heatmap_fij_2,
        'profile_ligne_graph': profile_ligne_graph,
        'profile_colonne_graph': profile_colonne_graph,
    })
