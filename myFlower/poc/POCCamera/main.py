import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2
from PIL import ImageTk
from PIL import Image


def open_file():
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    txt_edit.delete(1.0, tk.END)
    with open(filepath, "r") as input_file:
        text = input_file.read()
        txt_edit.insert(tk.END, text)
    window.title(f"Simple Text Editor - {filepath}")


def save_file():
    """Save the current file as a new file."""
    filepath = asksaveasfilename(
        defaultextension="txt",
        filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
    )
    if not filepath:
        return
    with open(filepath, "w") as output_file:
        text = txt_edit.get(1.0, tk.END)
        output_file.write(text)
    window.title(f"Simple Text Editor - {filepath}")


def screenshot():
    """ Fonction qui affiche le feed de la caméra de l'utilisateur dans un "Label" tkinter, cette fonction
     est récursive, à changer plus tard, éventuellement jouer avec des threads, à voir"""
    _, frame = video.read()
    fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(fr, "RGB")
    i = ImageTk.PhotoImage(image=img)
    LB_Camera.configure(image=i)
    LB_Camera.image = i
    LB_Camera.after(1, screenshot)

""" Création de la fenêtre principale de l'UI"""
window = tk.Tk()
window.title("Test d'application")
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

txt_edit = tk.Text(window)

""" Création d'un "frame" pour contenir les boutons de navigations """
FR_buttons = tk.Frame(window, relief=tk.RAISED, bd=2, background="red", width=450, height=50)
FR_buttons.pack(side="top", fill="x", expand=0)

""" Création d'un "Frame" pour contenir les informations, seulement la caméra actuellement mais plus tard 
aussi la collection de photo etc ..."""
FR_camera = tk.Frame(window, relief=tk.FLAT, bd=5, background="green")
FR_camera.pack(side="bottom", fill="both", expand=1)

""" Création d'un "Label" qui contiendra ensuite le flux de la caméra de l'utilisateur, photo en placeholder"""
image = tk.PhotoImage(file='images/fleur.png')
LB_Camera = tk.Label(FR_camera, image=image)
LB_Camera.pack()

""" Créations des boutons du menu supérieur, pas encore fonctionnel, le premier permet cependant
d'ouvrir le gestionnaire de fichier et de charger un .txt. Le deuxième permet "d'enregistrer sous" """
BTN_open = tk.Button(FR_buttons, text="Charger photo", command=open_file)
BTN_open.pack(side="left", fill="x", expand=1)

BTN_save = tk.Button(FR_buttons, text="Sauvegarder photo", command=save_file)
BTN_save.pack(side="left", fill="x", expand=1)

""" Lancement de la capture vidéo """
video = cv2.VideoCapture(0)

""" Lancement de la fonction permettant l'affichage du feed caméra sur le label LB_Camera"""
screenshot()

window.mainloop()
