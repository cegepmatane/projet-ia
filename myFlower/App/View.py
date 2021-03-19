import tkinter as Tk
from datetime import datetime

from tkscrolledframe import ScrolledFrame


class View(object):

    def __init__(self):

        print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Initialisation de View()")
        self.frame = None
        self.image = None
        self.classification = None
        self.classifications = None
        self.label = None
        self.camera = None
        self.upload = None
        print("     ",datetime.now().strftime("%Hh%Mm%Ss"), ": View() initialisée.")

    def get_structure(self, change_view):

        root = Tk.Tk()
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=9)
        root.grid_columnconfigure(0, weight=3)

        fr_header = Tk.Frame(root, relief=Tk.FLAT, background="orange")
        fr_header.grid(row=0, column=0, sticky="nswe")
        fr_header.grid_rowconfigure(0, weight=1)
        fr_header.grid_columnconfigure(0, weight=1)
        fr_header.grid_columnconfigure(1, weight=1)

        bt_home = Tk.Button(fr_header, text="Accueil", command=lambda: change_view("Accueil")
                            , bg="#5B5B5B", fg="#f87e28", font=("Courier", 33))
        bt_home.grid(row=0, column=0, sticky="nswe")

        bt_collection = Tk.Button(fr_header, text="Collection", command=lambda: change_view("Collection")
                                  , bg="#5B5B5B", fg="#f87e28", font=("Courier", 33))
        bt_collection.grid(row=0, column=1, sticky="nswe")
        return root

    def get_home(self, root, askopenfile, faire_classification_webcam):
        print("        ",datetime.now().strftime("%Hh%Mm%Ss"),": Chargement de la vue accueil...")

        self.frame = Tk.Frame(root, relief=Tk.FLAT, background="grey")

        self.label = Tk.Label(self.frame)
        self.label.place(relx=0.5, rely=0.5, anchor="center")

        self.upload = Tk.PhotoImage(file=r"image/upload.png").subsample(12, 12)
        bt_importer_image = Tk.Button(self.frame, image=self.upload, command=askopenfile, bg="#5b5b5b")
        bt_importer_image.place(relx=0.5, rely=0.9, anchor="center")

        self.camera = Tk.PhotoImage(file=r"image/camera.png").subsample(9, 9)
        bt_classification_webcam = Tk.Button(self.frame, image=self.camera, command=faire_classification_webcam,
                                             bg="#5b5b5b")
        bt_classification_webcam.place(relx=0.5, rely=0.8, anchor="center")

        self.frame.grid(row=1, column=0, sticky="nswe")

        print("        ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue accueil chargée.")

        return self.frame, self.label

    def get_collection(self, root, change_view, classifications):
        print("        ",datetime.now().strftime("%Hh%Mm%Ss"),": Chargement de la vue collection...")

        sf_page = ScrolledFrame(root)
        sf_page.grid(row=1, column=0, sticky="nswe")

        self.frame = sf_page.display_widget(Tk.Frame)

        # Grid
        ligne = 0
        colonne = 0
        self.classifications = classifications
        for classification in classifications:
            # Conteneur
            if ligne > 5:
                colonne = colonne + 1
                ligne = 0
            fr_fleur = Tk.Frame(self.frame, background="white")
            fr_fleur.grid(row=colonne, column=ligne, padx=5, pady=5)

            # Nom
            lb_nom_fleur = Tk.Label(fr_fleur, text=classification.get_type(), font=(None, 15))
            lb_nom_fleur.grid(row=0, column=1, padx=5, pady=5)
            # Image
            bt_image = Tk.Button(fr_fleur, image=classification.get_miniature(),
                                 command=lambda c=classification.get_id(): change_view("Details", id_classification=c))
            bt_image.grid(row=0, rowspan=2, column=0, padx=5, pady=5)
            # Date
            lb_date_fleur = Tk.Label(fr_fleur, text=classification.get_date(), font=(None, 9))
            lb_date_fleur.grid(row=1, column=1, padx=5, pady=5)

            ligne = ligne + 1

        print("        ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue collection chargée.")

        return self.frame

    def get_details(self, root, classification, supprimer_classification, modifier_note):
        print("        ",datetime.now().strftime("%Hh%Mm%Ss"),": Chargement de la vue détails...")

        self.classification = classification

        self.frame = Tk.Frame(root, relief=Tk.FLAT, background="white")
        self.frame.grid(row=1, column=0, sticky="nswe")

        cv_image = Tk.Canvas(self.frame, width=root.winfo_screenwidth(),
                             height=root.winfo_screenheight())
        cv_image.pack()

        # Ajout de l'image
        cv_image.create_image(0, 0, image=self.classification.get_image(), anchor="nw")

        # Ajout du texte indiquant le type de fleur
        txt_type_de_plante = cv_image.create_text(100, root.winfo_screenheight() * 0.6, anchor=Tk.NW,
                                                  fill="#f87e28", font=("Courier", 33))
        cv_image.itemconfig(txt_type_de_plante, text=self.classification.get_type())

        # Ajout du texte indiquant la date de classification
        txt_date_de_classification = cv_image.create_text(100, root.winfo_screenheight() * 0.75, anchor=Tk.NW,
                                                          fill="#f87e28", font=("Courier", 33))
        cv_image.itemconfig(txt_date_de_classification, text=self.classification.get_date())

        # Création d'une "Frame" pour contenir la zone de texte contenant la note
        fr_note = Tk.Frame(cv_image, width=int(root.winfo_screenwidth() * 0.20),
                           height=int(root.winfo_screenheight() * 0.10))
        fr_note.place(relx=0.8, rely=0.8, anchor="center")
        fr_note.pack_propagate(False)

        # Ajout de la zone de texte dans la "Frame" précédemment créée
        txt_note = Tk.Text(fr_note, font=("Courier", 12), bg="#5b5b5b", fg="#f87e28")
        txt_note.insert(Tk.INSERT, self.classification.get_note())
        txt_note.bind('<Return>', lambda event, a=classification.get_id():modifier_note(a, txt_note.get("1.0", Tk.END)))

        txt_note.pack()

        # Ajout du bouton pour supprimer une classification
        bt_supprimer_photo = Tk.Button(cv_image, text="Supprimer", bg="#5b5b5b", fg="#f87e28", font=("Courier", 22),
                                       command=lambda: supprimer_classification(classification.get_id()))
        bt_supprimer_photo.place(relx=0.1, rely=0.1, anchor="center")

        print("        ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue détails chargée.")

        return self.frame
