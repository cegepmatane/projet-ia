import tkinter as tk
import tkinter.font as font


def ouvrir_accueil():
    """Ouvre la page accueil"""
    print("Ouverture d'accueil")


def ouvrir_collection():
    """Ouvre la page collection"""
    print("Ouverture de collection")

def supprimer_classification():
    """Supprime la classification"""
    print("Suppression de la classification")


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

# Ajout du texte indiquant le type de fleur
txt_type_de_plante = cv_image.create_text(100, window.winfo_screenheight() * 0.7, anchor=tk.NW, fill="green")
cv_image.itemconfig(txt_type_de_plante, text="Type de fleur : Tournesol", font=fontBouton)

# Ajout du texte indiquant la date de classification
txt_date_de_classification = cv_image.create_text(100, window.winfo_screenheight() * 0.75, anchor=tk.NW, fill="orange")
cv_image.itemconfig(txt_date_de_classification, text="Analyse faite le : 24/01/2021", font=fontBouton)

# Création d'une "Frame" pour contenir la zone de texte contenant la note
fr_note = tk.Frame(cv_image, width=int(window.winfo_screenwidth() * 0.20), height=int(window.winfo_screenheight() * 0.10))
fr_note.place(x=int(window.winfo_screenwidth() * 0.70), y=int(window.winfo_screenheight() * 0.70))
fr_note.pack_propagate(False)

# Ajout de la zone de texte dans la "Frame" précédemment créée
txt_note = tk.Text(fr_note)
txt_note.insert(tk.INSERT, "Note : Trouvée sur la montagne au sud de Matane.")
txt_note.pack()

# Ajout du bouton pour supprimer une classification
bt_importer_photo = tk.Button(cv_image, text="Supprimer", font=fontBouton, command=supprimer_classification, bg="#f87e28", fg="#5b5b5b")
bt_importer_photo.place(x=int(window.winfo_screenwidth() * 0.05), y=int(window.winfo_screenheight() * 0.05))

# Créations des boutons du menu supérieur, pas encore fonctionnel, le premier servira a aller sur la page accueil
# et le deuxième à aller sur la page collection, tout deux mis dans la "Frame" fr_menu
btn_accueil = tk.Button(fr_menu, text="Accueil", command=ouvrir_accueil, bg="#5b5b5b", fg="#f87e28")
btn_accueil['font'] = fontBouton
btn_accueil.pack(side="left", fill="x", expand=1)
btn_collection = tk.Button(fr_menu, text="Collection", command=ouvrir_collection, bg="#5b5b5b", fg="#f87e28")
btn_collection['font'] = fontBouton
btn_collection.pack(side="left", fill="x", expand=1)

window.mainloop()
