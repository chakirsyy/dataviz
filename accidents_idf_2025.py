"""
TP Dataviz - Accidents corporels de la circulation en Île-de-France, année 2025.

Source : bases BAAC (ONISR), data.gouv.fr
  - caract-2025.csv  : une ligne par accident (date, heure, luminosité, département, coordonnées)
  - usagers-2025.csv : une ligne par usager impliqué (gravité des blessures)

Sorties (dans le dossier du script) :
  - carte_accidents_idf_2025.png
  - luminosite_gravite_idf_2025.png
  - stats_idf_2025.txt (chiffres repris dans le .md)

Usage : python accidents_idf_2025.py
Les CSV sont lus dans ./data/ ; s'ils sont absents, ils sont téléchargés.
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

ICI = Path(__file__).resolve().parent
DATA = ICI / "data"

BAAC = ("https://static.data.gouv.fr/resources/bases-de-donnees-annuelles-des-"
        "accidents-corporels-de-la-circulation-routiere-annees-de-2005-a-2025/")
SOURCES = {
    "caract-2025.csv": BAAC + "20261030-141517/caract-2025.csv",
    "usagers-2025.csv": BAAC + "20261030-141441/usagers-2025.csv",
    "departements.geojson": ("https://raw.githubusercontent.com/gregoiredavid/"
                             "france-geojson/master/departements.geojson"),
}

IDF = {"75": "Paris", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
       "94": "Val-de-Marne", "77": "Seine-et-Marne", "78": "Yvelines",
       "91": "Essonne", "95": "Val-d'Oise"}
ZONE = {"75": "Paris", "92": "Petite couronne", "93": "Petite couronne",
        "94": "Petite couronne", "77": "Grande couronne", "78": "Grande couronne",
        "91": "Grande couronne", "95": "Grande couronne"}

# Luminosité BAAC -> 4 modalités (règle 11 : les deux cas de nuit sans lumière sont fusionnés)
LUM = {"1": "Plein jour", "2": "Aube ou crépuscule", "3": "Nuit sans éclairage",
       "4": "Nuit sans éclairage", "5": "Nuit avec éclairage"}

# Gravité usager BAAC : 1 indemne, 2 tué, 3 blessé hospitalisé, 4 blessé léger
RANG_GRAV = {1: 0, 4: 1, 3: 2, 2: 3}
CLASSE = {0: "Sans blessé", 1: "Léger", 2: "Hospitalisé", 3: "Mortel"}

# Palette (bleu séquentiel + orange d'accent, couple validé daltonisme)
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
MUET = "#898781"
TRAIT = "#c3c2b7"
ORANGE = "#eb6834"
BLEU = "#2a78d6"
GRIS_BARRE = "#c9c8c1"
RAMPE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "font.size": 11,
    "text.color": ENCRE,
    "axes.labelcolor": ENCRE_2,
    "xtick.color": ENCRE_2,
    "ytick.color": ENCRE_2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def entier(n):
    return f"{int(round(n)):,}".replace(",", " ")


def pct(x):
    return f"{x:.1f}".replace(".", ",") + " %"


def chemin(nom):
    p = DATA / nom
    if not p.exists():
        DATA.mkdir(exist_ok=True)
        print(f"Téléchargement de {nom}...")
        urllib.request.urlretrieve(SOURCES[nom], p)
    return p


def lire_baac(nom):
    for enc in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(chemin(nom), sep=";", dtype=str, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    df.columns = df.columns.str.strip().str.replace('"', "")
    # Depuis 2022 la colonne s'appelle Accident_Id dans le fichier caractéristiques
    return df.rename(columns={"Accident_Id": "Num_Acc"})


# --------------------------------------------------------------------------
# 1. Chargement et nettoyage
# --------------------------------------------------------------------------
journal = []


def log(msg):
    print(msg)
    journal.append(msg)


caract = lire_baac("caract-2025.csv")
usagers = lire_baac("usagers-2025.csv")
log(f"Accidents France 2025 : {entier(len(caract))}")
log(f"Usagers France 2025 : {entier(len(usagers))}")

caract["dep"] = caract["dep"].str.strip()
acc = caract[caract["dep"].isin(IDF)].copy()
log(f"Accidents déclarés en Île-de-France : {entier(len(acc))}")

for col in ("lat", "long"):
    acc[col] = pd.to_numeric(acc[col].str.strip().str.replace(",", "."), errors="coerce")
sans_coord = acc["lat"].isna() | acc["long"].isna() | (acc["lat"] == 0) | (acc["long"] == 0)
log(f"  sans coordonnées exploitables : {entier(sans_coord.sum())}")
acc = acc[~sans_coord]

deps = gpd.read_file(chemin("departements.geojson"))
deps = deps[deps["code"].isin(IDF)].to_crs(2154)

pts = gpd.GeoDataFrame(acc, geometry=gpd.points_from_xy(acc["long"], acc["lat"]), crs=4326).to_crs(2154)
pts = gpd.sjoin(pts, deps[["code", "geometry"]], how="left", predicate="within")
pts = pts[~pts.index.duplicated()]
incoherent = pts["code"] != pts["dep"]
log(f"  coordonnées hors du département déclaré : {entier(incoherent.sum())}")
pts = pts[~incoherent].drop(columns=["index_right", "code"])

pts["luminosite"] = pts["lum"].str.strip().map(LUM)
pts["zone"] = pts["dep"].map(ZONE)

usagers["grav"] = pd.to_numeric(usagers["grav"].str.strip(), errors="coerce")
gravite = (usagers.assign(rang=usagers["grav"].map(RANG_GRAV))
           .groupby("Num_Acc")["rang"].max())
pts["rang"] = pts["Num_Acc"].map(gravite)
log(f"  sans gravité renseignée : {entier(pts['rang'].isna().sum())}")
pts = pts.dropna(subset=["rang"])
pts["gravite"] = pts["rang"].astype(int).map(CLASSE)
pts["grave"] = pts["rang"] >= 2  # au moins un tué ou un blessé hospitalisé

log(f"Accidents retenus après nettoyage : {entier(len(pts))}")
log(f"  dont mortels : {entier((pts['gravite'] == 'Mortel').sum())}, "
    f"graves (tué ou hospitalisé) : {entier(pts['grave'].sum())}")

# Tableau par département
par_dep = (pts.groupby("dep")
           .agg(accidents=("Num_Acc", "size"),
                mortels=("gravite", lambda s: (s == "Mortel").sum()),
                graves=("grave", "sum"))
           .assign(part_graves=lambda d: 100 * d["graves"] / d["accidents"],
                   nom=lambda d: d.index.map(IDF))
           .sort_values("accidents", ascending=False))
log("\nPar département :\n" + par_dep.to_string())

par_zone = (pts.groupby("zone")
            .agg(accidents=("Num_Acc", "size"),
                 mortels=("gravite", lambda s: (s == "Mortel").sum()),
                 graves=("grave", "sum"))
            .assign(part_accidents=lambda d: 100 * d["accidents"] / d["accidents"].sum(),
                    part_mortels=lambda d: 100 * d["mortels"] / d["mortels"].sum(),
                    part_graves=lambda d: 100 * d["graves"] / d["accidents"]))
log("\nPar zone :\n" + par_zone.to_string())

# --------------------------------------------------------------------------
# 2. Carte : densité des accidents (hexbin, règle 13) + accidents mortels
#    Gauche : toute la région, hexagones de 2 km
#    Droite : zoom Paris + petite couronne, hexagones de 500 m
#    Une seule échelle (accidents par km²) pour les deux panneaux
# --------------------------------------------------------------------------
cmap = LinearSegmentedColormap.from_list("bleus", RAMPE)
mortels = pts[pts["gravite"] == "Mortel"]
centre_codes = ["75", "92", "93", "94"]


def hexbin_densite(ax, geo, largeur, bornes):
    """Hexagones réguliers de `largeur` mètres, valeur = accidents par km²."""
    xmin, ymin, xmax, ymax = bornes
    nx = max(1, int(np.ceil((xmax - xmin) / largeur)))
    xmax = xmin + nx * largeur
    sy = np.sqrt(3) * largeur
    ny = max(1, int(np.ceil((ymax - ymin) / sy)))
    ymax = ymin + ny * sy
    hb = ax.hexbin(geo.x, geo.y, gridsize=(nx, ny), extent=(xmin, xmax, ymin, ymax),
                   mincnt=1, cmap=cmap, linewidths=0.2, edgecolors=SURFACE, zorder=1)
    v = hb.get_paths()[0].vertices
    aire = 0.5 * abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) - np.dot(v[:, 1], np.roll(v[:, 0], 1))) / 1e6
    hb.set_array(hb.get_array() / aire)
    return hb


def etiquette(ax, d, taille, dx=0, dy=0):
    p = d.geometry.representative_point()
    r = par_dep.loc[d["code"]]
    ax.text(p.x + dx, p.y + dy,
            f"{IDF[d['code']]}\n{entier(r['accidents'])} accidents · {int(r['mortels'])} mortels",
            ha="center", va="center", fontsize=taille, color=ENCRE, zorder=5, linespacing=1.3,
            path_effects=[pe.withStroke(linewidth=3.2, foreground=SURFACE)])


def fond(ax, polys):
    polys.plot(ax=ax, facecolor="#f3f2ee", edgecolor="none", zorder=0)


def contours(ax, polys):
    polys.boundary.plot(ax=ax, color=ENCRE_2, linewidth=0.8, zorder=2)


def points_mortels(ax, taille):
    ax.scatter(mortels.geometry.x, mortels.geometry.y, s=taille, color=ORANGE,
               edgecolor="white", linewidth=0.8, zorder=3)


fig = plt.figure(figsize=(16, 8.6))
ax_r = fig.add_axes([0.0, 0.1, 0.56, 0.78])
ax_z = fig.add_axes([0.58, 0.1, 0.42, 0.78])

# Panneau région
bornes_r = deps.total_bounds
fond(ax_r, deps)
hb_r = hexbin_densite(ax_r, pts.geometry, 2000, bornes_r)
contours(ax_r, deps)
points_mortels(ax_r, 18)
decal_r = {"95": (0, 6000), "78": (-7000, 0)}
for _, d in deps[~deps["code"].isin(centre_codes)].iterrows():
    etiquette(ax_r, d, 10.5, *decal_r.get(d["code"], (0, 0)))

centre = deps[deps["code"].isin(centre_codes)]
cx0, cy0, cx1, cy1 = centre.total_bounds
marge = 1500
cadre = (cx0 - marge, cy0 - marge, cx1 + marge, cy1 + marge)
ax_r.add_patch(plt.Rectangle((cadre[0], cadre[1]), cadre[2] - cadre[0], cadre[3] - cadre[1],
                             fill=False, edgecolor=ENCRE, linewidth=1.1, zorder=4))
ax_r.text(cadre[2], cadre[1] - 1200, "zoom à droite", fontsize=9.5, color=ENCRE_2, ha="right", va="top",
          path_effects=[pe.withStroke(linewidth=3, foreground=SURFACE)], zorder=5)

# Panneau zoom Paris + petite couronne
x, y = pts.geometry.x, pts.geometry.y
dans_cadre = pts[(x >= cadre[0]) & (x <= cadre[2]) & (y >= cadre[1]) & (y <= cadre[3])]
fond(ax_z, deps)
hb_z = hexbin_densite(ax_z, dans_cadre.geometry, 500, cadre)
contours(ax_z, deps)
points_mortels(ax_z, 34)
decal_z = {"92": (-1500, 0)}
for _, d in centre.iterrows():
    etiquette(ax_z, d, 11, *decal_z.get(d["code"], (0, 0)))

# Échelle commune aux deux panneaux
valeurs = np.concatenate([hb_r.get_array(), hb_z.get_array()])
norme = LogNorm(vmin=valeurs.min(), vmax=valeurs.max())
for hb in (hb_r, hb_z):
    hb.set_norm(norme)

for ax in (ax_r, ax_z):
    ax.set_axis_off()
    ax.set_aspect("equal")
ax_z.set_xlim(cadre[0], cadre[2])
ax_z.set_ylim(cadre[1], cadre[3])

cax = fig.add_axes([0.02, 0.085, 0.26, 0.016])
cb = fig.colorbar(hb_r, cax=cax, orientation="horizontal")
graduations = [v for v in (0.3, 1, 3, 10, 30, 100, 300, 1000) if norme.vmin <= v <= norme.vmax]
cb.locator = FixedLocator(graduations)
cb.formatter = FuncFormatter(lambda v, _: str(v).replace(".", ",") if v < 1 else entier(v))
cb.update_ticks()
cb.minorticks_off()
cb.outline.set_visible(False)
cb.ax.tick_params(labelsize=9.5, length=0)
cb.set_label("Accidents par km² (hexagones de 2 km à gauche, 500 m à droite)", fontsize=9.5, color=ENCRE_2)

fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=8,
                           markerfacecolor=ORANGE, markeredgecolor="white",
                           label=f"Accident mortel ({entier(len(mortels))})")],
           loc="lower left", bbox_to_anchor=(0.32, 0.07), frameon=False, fontsize=10.5)

zp = par_zone
fig.text(0.02, 0.975, "Accidents corporels de la circulation en Île-de-France, 2025",
         fontsize=17, fontweight="semibold", ha="left", va="top")
fig.text(0.02, 0.93,
         f"Paris et la petite couronne concentrent "
         f"{pct(zp.loc[['Paris', 'Petite couronne'], 'part_accidents'].sum())} des accidents, "
         f"mais la grande couronne compte {pct(zp.loc['Grande couronne', 'part_mortels'])} des accidents mortels",
         fontsize=12, color=ENCRE_2, ha="left", va="top")
fig.text(0.99, 0.085, f"Source : BAAC 2025 (ONISR), data.gouv.fr · {entier(len(pts))} accidents géolocalisés",
         fontsize=9, color=MUET, ha="right")
fig.savefig(ICI / "carte_accidents_idf_2025.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 3. Comparaison selon la luminosité, globale puis par zone
#    (la luminosité est liée au lieu : la nuit sans éclairage, c'est surtout la grande couronne)
# --------------------------------------------------------------------------
ZONES = ["Paris", "Petite couronne", "Grande couronne"]
COUL_ZONE = dict(zip(ZONES, ["#86b6ef", "#2a78d6", "#104281"]))  # rampe ordinale, centre -> périphérie

avec_lum = pts.dropna(subset=["luminosite"])
lum = (avec_lum.groupby("luminosite")
       .agg(accidents=("Num_Acc", "size"), graves=("grave", "sum"))
       .assign(part=lambda d: 100 * d["graves"] / d["accidents"])
       .sort_values("part"))  # règle 12 : tri, la plus grave en haut
lum_zone = (avec_lum.groupby(["luminosite", "zone"])
            .agg(accidents=("Num_Acc", "size"), graves=("grave", "sum"))
            .assign(part=lambda d: 100 * d["graves"] / d["accidents"]))
log("\nPar luminosité :\n" + lum.to_string())
log("\nPar luminosité et zone :\n" + lum_zone.to_string())

pire = lum["part"].idxmax()
part_gc_pire = 100 * lum_zone.loc[(pire, "Grande couronne"), "accidents"] / lum.loc[pire, "accidents"]
gc = lum_zone.xs("Grande couronne", level="zone")["part"]
log(f"\n{pire} : {pct(part_gc_pire)} des accidents en grande couronne ; "
    f"grande couronne de {pct(gc.min())} à {pct(gc.max())} selon la luminosité")

fig = plt.figure(figsize=(12, 5.4))
a1 = fig.add_axes([0.2, 0.15, 0.33, 0.53])
a2 = fig.add_axes([0.6, 0.15, 0.36, 0.53], sharey=a1)
y = np.arange(len(lum))
xmax = max(lum["part"].max(), lum_zone["part"].max()) * 1.15  # même échelle pour les deux panneaux

# Panneau gauche : toute la région
a1.barh(y, lum["part"], color=GRIS_BARRE, height=0.6)
for i, v in enumerate(lum["part"]):
    a1.text(v + xmax * 0.015, i, pct(v), va="center", fontsize=10.5,
            fontweight="semibold" if lum.index[i] == pire else "normal")
a1.set_yticks(y, [f"{c}\n{entier(n)} accidents" for c, n in zip(lum.index, lum["accidents"])],
              fontsize=10.5)

# Panneau droit : une pastille par zone
for i, c in enumerate(lum.index):
    v = lum_zone.loc[c, "part"].reindex(ZONES)
    a2.plot([v.min(), v.max()], [i, i], color=TRAIT, linewidth=1.5, zorder=1)
    for z in ZONES:
        a2.scatter(v[z], i, s=80, color=COUL_ZONE[z], edgecolor="white", linewidth=1.2, zorder=2)
    if i == len(lum) - 1:  # étiquetage direct sur la première ligne (règle 9)
        a2.text(v["Paris"] - xmax * 0.02, i, "Paris", ha="right", va="center", fontsize=9.5, color=ENCRE_2)
        for z in ZONES[1:]:
            a2.text(v[z], i + 0.25, z, ha="center", va="bottom", fontsize=9.5, color=ENCRE_2)
for i, c in enumerate(lum.index):
    v = lum_zone.loc[(c, "Grande couronne"), "part"]
    a2.text(v + xmax * 0.025, i, pct(v), va="center", fontsize=9.5, color=ENCRE_2)
a2.tick_params(axis="y", labelleft=False)

for a, titre in ((a1, "Toute l'Île-de-France"), (a2, "Par zone")):
    a.set_xlim(0, xmax)
    a.set_ylim(-0.5, len(lum) - 0.2)
    a.set_title(titre, loc="left", fontsize=11.5, color=ENCRE, pad=8)
    a.set_xticks([])
    for cote in ("top", "right", "bottom"):
        a.spines[cote].set_visible(False)
    a.spines["left"].set_color(TRAIT)
    a.tick_params(axis="y", length=0)

fig.legend(handles=[Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=COUL_ZONE[z],
                           markeredgecolor="white", label=z) for z in ZONES],
           loc="lower left", bbox_to_anchor=(0.6, 0.04), ncol=3, frameon=False, fontsize=9.5,
           handletextpad=0.3, columnspacing=1.2)

jour = lum.loc["Plein jour", "part"]
fig.text(0.02, 0.97, "Gravité des accidents selon la luminosité, Île-de-France 2025",
         fontsize=15, fontweight="semibold", ha="left", va="top")
fig.text(0.02, 0.905,
         f"La nuit sans éclairage paraît plus dangereuse ({pct(lum.loc[pire, 'part'])} d'accidents graves contre "
         f"{pct(jour)} en plein jour), mais l'écart tient surtout au lieu :\n"
         f"{pct(part_gc_pire)} de ces accidents ont lieu en grande couronne, où un accident sur cinq environ "
         f"est grave quelle que soit la lumière.",
         fontsize=11, color=ENCRE_2, ha="left", va="top", linespacing=1.4)
fig.text(0.2, 0.775, "Part d'accidents graves : au moins un tué ou un blessé hospitalisé",
         fontsize=10, color=MUET, ha="left")
n_paris = lum_zone.loc[(pire, "Paris"), "accidents"]
fig.text(0.02, 0.0,
         f"Source : BAAC 2025 (ONISR), data.gouv.fr · luminosité non renseignée exclue · "
         f"Paris, nuit sans éclairage : {entier(n_paris)} accidents seulement, taux peu fiable",
         fontsize=9, color=MUET, ha="left")
fig.savefig(ICI / "luminosite_gravite_idf_2025.png", dpi=150, bbox_inches="tight")
plt.close(fig)

(ICI / "stats_idf_2025.txt").write_text("\n".join(journal), encoding="utf-8")
print("\nFigures enregistrées.")
