import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import cv2
from PIL import ImageTk
from PIL import Image
import random
from threading import Thread
from threading import Lock


class WebcamVideoStream:

    def __init__(self):
        self.stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        (self.grabbed, self.frame) = self.stream.read()
        self.thread = Thread(target=self.update, args=())
        self.started = False
        self.read_lock = Lock()

    def start(self):
        if self.started:
            print ("WebcamVideoStream already started")
            return None
        self.started = True
        if self.thread.is_alive():
            print ("WebcamVideoStream thread still alive")
        else:
            print(self.thread.is_alive())
            self.thread.start()
        return self

    def update(self):
        while self.started:
            (grabbed, frame) = self.stream.read()
            self.read_lock.acquire()
            self.grabbed, self.frame = grabbed, frame
            self.read_lock.release()

    def read(self):
        self.read_lock.acquire()
        frame = self.frame.copy()
        self.read_lock.release()
        return frame

    def stop(self):
        self.started = False
        self.stream.release()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stream.release()


class WebcamVideoRead:

    def __init__(self, wvs):
        self.wvs = wvs
        self.started = False
        self.thread = Thread(target=self.update, args=())

    def start(self):
        self.started = True
        self.thread.start()

    def update(self):
        while self.started:
            fr = cv2.cvtColor(self.wvs.read(), cv2.COLOR_BGR2RGB)
            img = Image.fromarray(fr, "RGB")
            i = ImageTk.PhotoImage(image=img)
            LB_Camera.configure(image=i)
            LB_Camera.image = i

    def stop(self):
        self.started = False


def open_file():
    vr.stop()
    vs.stop()
    """Open a file for editing."""
    filepath = askopenfilename(
        filetypes=[("Image File", "*.png"), ("All Files", "*.*")]
    )
    if not filepath:
        return
    print(filepath)

    print("Locked") if vs.read_lock.locked() else print("Unlocked")

    i = ImageTk.PhotoImage(file=filepath)
    LB_Camera.configure(image=i)
    LB_Camera.image = i


def save_file():
    vs.start()
    vr.start()


""" Création de la fenêtre principale de l'UI"""
window = tk.Tk()
window.title("Test d'application")
window.rowconfigure(0, minsize=800, weight=1)
window.columnconfigure(1, minsize=800, weight=1)

""" Création d'un "frame" pour contenir les boutons de navigations """
FR_buttons = tk.Frame(window, relief=tk.RAISED, bd=2, background="red", width=450, height=50)
FR_buttons.pack(side="top", fill="x", expand=0)

""" Création d'un "Frame" pour contenir les informations, seulement la caméra actuellement mais plus tard 
aussi la collection de photo etc ..."""
FR_camera = tk.Frame(window, relief=tk.FLAT, bd=5, background="#909090")
FR_camera.pack(side="bottom", fill="both", expand=1)

""" Création d'un "Label" qui contiendra ensuite le flux de la caméra de l'utilisateur, photo en placeholder"""
image = tk.PhotoImage(file='./Images/fleur.png')
LB_Camera = tk.Label(FR_camera, image=image)
LB_Camera.pack()

""" Créations des boutons du menu supérieur, pas encore fonctionnel, le premier permet cependant
d'ouvrir le gestionnaire de fichier et de charger un .txt. Le deuxième permet "d'enregistrer sous" """
BTN_open = tk.Button(FR_buttons, text="Charger photo", command=open_file)
BTN_open.pack(side="left", fill="x", expand=1)

BTN_save = tk.Button(FR_buttons, text="Sauvegarder photo", command=save_file)
BTN_save.pack(side="left", fill="x", expand=1)

""" Lancement de la capture vidéo """

""" Lancement de la fonction permettant l'affichage du feed caméra sur le label LB_Camera"""

vs = WebcamVideoStream().start()

vr = WebcamVideoRead(vs)
vr.start()

window.mainloop()