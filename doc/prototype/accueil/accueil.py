import tkinter as tk
import tkinter.font as font


def ouvrir_accueil():
    """Ouvre la page accueil"""
    print("Ouverture d'accueil")


def ouvrir_collection():
    """Ouvre la page collection"""
    print("Ouverture de collection")


def prendre_photo():
    """Prends une photo"""
    print("Prise de photo")

    
def importer_photo():
    """Importe une photo"""
    print("Importation de photo")


# Création de la fenêtre principale de l'UI
window = tk.Tk()
window.title("Prototype d'interface")

# Création d'une font ayant une taille plus grande pour les éléments importants
fontBouton = font.Font(size=15)

# Création d'un "frame" pour contenir les boutons de navigations
fr_menu = tk.Frame(window, relief=tk.FLAT, bd=0, background="orange", width=450, height=50)
fr_menu.pack(side="top", fill="x", expand=0)

# Création d'un "Canvas" pour contenir la photo de la fleur
cv_image = tk.Canvas(window, width=window.winfo_screenwidth() - fr_menu.winfo_width(), height=window.winfo_screenheight() - fr_menu.winfo_height())
cv_image.pack()

# Ajout de l'image
image = tk.PhotoImage(file='./image/fleur.png')
cv_image.create_image(0, 0, image=image, anchor="nw")

# Ajout du bouton pour prendre une photo
icone_photo = tk.PhotoImage(file='./image/camera.png')
icone_photo = tk.PhotoImage.subsample(icone_photo, 8, 8)
bt_prendre_photo = tk.Button(cv_image, image = icone_photo, command=prendre_photo, bg="#5b5b5b", fg="#f87e28")
bt_prendre_photo.place(x=int(window.winfo_screenwidth() * 0.4725), y=int(window.winfo_screenheight() * 0.65))

# Ajout du bouton pour importer une photo
bt_importer_photo = tk.Button(cv_image, text="+", font=fontBouton, command=prendre_photo, bg="#5b5b5b", fg="#f87e28")
bt_importer_photo.place(x=int(window.winfo_screenwidth() * 0.485), y=int(window.winfo_screenheight() * 0.75))

# Créations des boutons du menu supérieur, pas encore fonctionnel, le premier servira a aller sur la page accueil
# et le deuxième à aller sur la page collection, tout deux mis dans la "Frame" fr_menu
btn_accueil = tk.Button(fr_menu, text="Accueil", command=ouvrir_accueil, bg="#5b5b5b", fg="#f87e28")
btn_accueil['font'] = fontBouton
btn_accueil.pack(side="left", fill="x", expand=1)
btn_collection = tk.Button(fr_menu, text="Collection", command=ouvrir_collection, bg="#5b5b5b", fg="#f87e28")
btn_collection['font'] = fontBouton
btn_collection.pack(side="left", fill="x", expand=1)

window.mainloop()
