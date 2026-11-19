# TP Dataviz - Accidents de la route en Île-de-France (2025)
# Données : base BAAC 2025 sur data.gouv.fr (caract-2025.csv et usagers-2025.csv)
# Contours des départements : https://github.com/gregoiredavid/france-geojson

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# ---------- Chargement ----------
# les deux CSV doivent être dans le même dossier que le script
caract = pd.read_csv("caract-2025.csv", sep=";", decimal=",", dtype={"dep": str})
usagers = pd.read_csv("usagers-2025.csv", sep=";")
deps = gpd.read_file("https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson")

print("Accidents en France :", len(caract))

# ---------- Nettoyage ----------
idf = ["75", "92", "93", "94", "77", "78", "91", "95"]
centre = ["75", "92", "93", "94"]  # Paris + petite couronne
noms = {"75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis", "94": "Val-de-Marne",
        "77": "Seine-et-Marne", "78": "Yvelines", "91": "Essonne", "95": "Val-d'Oise"}

acc = caract[caract["dep"].isin(idf)].copy()
print("Accidents en Île-de-France :", len(acc))

# on enlève les accidents sans coordonnées
acc = acc.dropna(subset=["lat", "long"])
acc = acc[(acc["lat"] != 0) & (acc["long"] != 0)]

# on passe en GeoDataFrame et en Lambert 93 (coordonnées en mètres)
acc = gpd.GeoDataFrame(acc, geometry=gpd.points_from_xy(acc["long"], acc["lat"]), crs="EPSG:4326")
acc = acc.to_crs("EPSG:2154")
deps = deps[deps["code"].isin(idf)].to_crs("EPSG:2154")

# on vérifie que chaque point est bien dans son département, sinon on le supprime
acc = gpd.sjoin(acc, deps[["code", "geometry"]], how="left", predicate="within")
print("Coordonnées hors du département :", (acc["code"] != acc["dep"]).sum())
acc = acc[acc["code"] == acc["dep"]]

# gravité usager : 1 indemne, 2 tué, 3 hospitalisé, 4 blessé léger
# on la remet dans l'ordre pour garder la victime la plus touchée de chaque accident
ordre = {1: 0, 4: 1, 3: 2, 2: 3}
usagers["niveau"] = usagers["grav"].map(ordre)
niveau_accident = usagers.groupby("Num_Acc")["niveau"].max()
acc["niveau"] = acc["Num_Acc"].map(niveau_accident)
acc["mortel"] = acc["niveau"] == 3
acc["grave"] = acc["niveau"] >= 2  # au moins un tué ou un hospitalisé

# luminosité : 5 catégories -> 4 (les deux cas de nuit sans lumière sont regroupés)
acc["luminosite"] = acc["lum"].map({1: "Plein jour", 2: "Aube ou crépuscule", 3: "Nuit sans éclairage",
                                    4: "Nuit sans éclairage", 5: "Nuit avec éclairage"})


def zone(dep):
    if dep == "75":
        return "Paris"
    elif dep in ["92", "93", "94"]:
        return "Petite couronne"
    else:
        return "Grande couronne"


acc["zone"] = acc["dep"].apply(zone)

print("Accidents gardés :", len(acc), "dont", acc["mortel"].sum(), "mortels et", acc["grave"].sum(), "graves")

# ---------- Quelques chiffres ----------
stats_zone = acc.groupby("zone").agg(accidents=("Num_Acc", "count"), mortels=("mortel", "sum"), graves=("grave", "sum"))
stats_zone["% des accidents"] = stats_zone["accidents"] / stats_zone["accidents"].sum() * 100
stats_zone["% des mortels"] = stats_zone["mortels"] / stats_zone["mortels"].sum() * 100
stats_zone["% graves"] = stats_zone["graves"] / stats_zone["accidents"] * 100
print(stats_zone.round(1))

stats_dep = acc.groupby("dep").agg(accidents=("Num_Acc", "count"), mortels=("mortel", "sum"))
print(stats_dep)

# ---------- Carte ----------
mortels = acc[acc["mortel"]]
acc_centre = acc[acc["dep"].isin(centre)]
xmin, ymin, xmax, ymax = deps[deps["code"].isin(centre)].total_bounds

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# à gauche : toute la région
deps.plot(ax=ax1, color="whitesmoke")
hb1 = ax1.hexbin(acc.geometry.x, acc.geometry.y, gridsize=70, bins="log", cmap="Blues", mincnt=1)
deps.boundary.plot(ax=ax1, color="grey", linewidth=0.8)
ax1.scatter(mortels.geometry.x, mortels.geometry.y, color="darkorange", s=10, label="Accident mortel")
ax1.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="black"))
ax1.set_title("Toute l'Île-de-France")
ax1.legend(loc="lower left")
fig.colorbar(hb1, ax=ax1, shrink=0.5, label="Nombre d'accidents par hexagone")

# à droite : zoom sur Paris et la petite couronne
deps.plot(ax=ax2, color="whitesmoke")
hb2 = ax2.hexbin(acc_centre.geometry.x, acc_centre.geometry.y, gridsize=50, bins="log", cmap="Blues", mincnt=1)
deps.boundary.plot(ax=ax2, color="grey", linewidth=0.8)
ax2.scatter(mortels.geometry.x, mortels.geometry.y, color="darkorange", s=20)
ax2.set_xlim(xmin, xmax)
ax2.set_ylim(ymin, ymax)
ax2.set_title("Zoom sur Paris et la petite couronne")
fig.colorbar(hb2, ax=ax2, shrink=0.5, label="Nombre d'accidents par hexagone")

# noms des départements directement sur la carte
for i, d in deps.iterrows():
    nb = stats_dep.loc[d["code"], "accidents"]
    nb_mortels = stats_dep.loc[d["code"], "mortels"]
    texte = noms[d["code"]] + "\n" + str(nb) + " accidents, " + str(nb_mortels) + " mortels"
    p = d.geometry.representative_point()
    if d["code"] in centre:
        ax = ax2
    else:
        ax = ax1
    ax.annotate(texte, (p.x, p.y), ha="center", va="center", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

ax1.axis("off")
ax2.axis("off")

part_centre = stats_zone.loc[["Paris", "Petite couronne"], "% des accidents"].sum()
part_mortels_gc = stats_zone.loc["Grande couronne", "% des mortels"]
fig.suptitle(f"Accidents en Île-de-France (2025) : {part_centre:.0f} % des accidents sont à Paris et en petite couronne,\n"
             f"mais {part_mortels_gc:.0f} % des accidents mortels sont en grande couronne", fontsize=14)
fig.text(0.01, 0.01, "Source : BAAC 2025, data.gouv.fr", fontsize=8, color="grey")
plt.tight_layout()
plt.savefig("carte_accidents_idf_2025.png", dpi=150)

# ---------- Gravité selon la luminosité ----------
# part d'accidents graves pour chaque luminosité, triée (la plus grave en haut)
lum = (acc.groupby("luminosite")["grave"].mean() * 100).sort_values()
lum_zone = acc.groupby(["luminosite", "zone"])["grave"].mean().unstack() * 100
lum_zone = lum_zone.loc[lum.index, ["Paris", "Petite couronne", "Grande couronne"]]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

barres = ax1.barh(lum.index, lum.values, color="grey")
ax1.bar_label(barres, fmt="%.1f %%", padding=3)
ax1.set_title("Toute l'Île-de-France")

lum_zone.plot(kind="barh", ax=ax2, color=["lightskyblue", "steelblue", "navy"], width=0.8)
for c in ax2.containers:
    ax2.bar_label(c, fmt="%.1f %%", padding=3, fontsize=8)
ax2.set_title("Par zone")
ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False)

for ax in [ax1, ax2]:
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

fig.suptitle("Part d'accidents graves (au moins un tué ou un hospitalisé) selon la luminosité, 2025\n"
             "La nuit sans éclairage semble plus grave, mais c'est surtout la grande couronne qui fait monter le chiffre",
             fontsize=12)
fig.text(0.01, 0.01, "Source : BAAC 2025, data.gouv.fr. Paris de nuit sans éclairage : seulement "
         + str(len(acc[(acc["zone"] == "Paris") & (acc["luminosite"] == "Nuit sans éclairage")])) + " accidents",
         fontsize=8, color="grey")
plt.tight_layout()
plt.savefig("luminosite_gravite_idf_2025.png", dpi=150)

print(lum.round(1))
print(lum_zone.round(1))
