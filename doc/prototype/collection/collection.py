from datetime import date
import tkinter as tk
import tkinter.font as font
from PIL import Image, ImageTk

# Fenêtre principale
window = tk.Tk()
window.title("Collection")
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)
window.configure(background="grey")

# Menu
fr_menu = tk.Frame(window, relief=tk.FLAT, bd=0, background="orange", width=450, height=50)
fr_menu.pack(side="top", fill="x", expand=0)
# Bouton accueil
bt_accueil = tk.Button(fr_menu, text="Accueil", command="", bg="#5b5b5b", fg="#f87e28")
bt_accueil['font'] = font.Font(size=15)
bt_accueil.pack(side="left", fill="x", expand=1)
# Bouton collection
bt_collection = tk.Button(fr_menu, text="Collection", command="", bg="#5b5b5b", fg="#f87e28")
bt_collection['font'] = font.Font(size=15)
bt_collection.pack(side="left", fill="x", expand=1)

# Page
fr_page = tk.Frame(window, background="grey")
fr_page.pack(side="bottom", fill="both", expand=True, padx=20, pady=10)

"""
# Image
image = tk.PhotoImage(file='./image/fleur.png')
lb_image = tk.Label(fr_page, image=image)
lb_image.pack()
"""

# Données simulées
fleurs = []
for ligne in range(41):
    image_fleur = Image.open('./image/fleur.png')
    largeur_de_base = 100
    pourcent_largeur = (largeur_de_base / float(image_fleur.size[0]))
    hauteur = int((float(image_fleur.size[1]) * float(pourcent_largeur)))
    image_fleur = image_fleur.resize((largeur_de_base, hauteur), Image.ANTIALIAS)
    image_fleur = ImageTk.PhotoImage(image_fleur)

    Object = lambda **kwargs: type("Object", (), kwargs)
    fleur = Object(nom="nom-fleur", image=image_fleur, date=date.today(),)
    fleurs.append(fleur)

# Grid
ligne = 0
colonne = 0
for fleur in fleurs:
    # Conteneur
    if ligne > 5:
        colonne = colonne + 1
        ligne = 0
    fr_fleur = tk.Frame(fr_page, background="white")
    fr_fleur.grid(row=colonne, column=ligne, padx=5, pady=5)

    # Nom
    lb_nom_fleur = tk.Label(fr_fleur, text=fleur.nom, font=(None, 15))
    lb_nom_fleur.grid(row=0, column=1, padx=5, pady=5)
    # Image
    lb_image = tk.Label(fr_fleur, image=image_fleur)
    lb_image.grid(row=0, rowspan=2, column=0, padx=5, pady=5)
    # Date
    lb_date_fleur = tk.Label(fr_fleur, text=fleur.date, font=(None, 9))
    lb_date_fleur.grid(row=1, column=1, padx=5, pady=5)

    ligne = ligne + 1

window.mainloop()
