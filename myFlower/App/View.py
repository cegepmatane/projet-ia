import tkinter as Tk

from tkscrolledframe import ScrolledFrame


class View(object):

    def __init__(self):
        print("Vue initialisée")
        self.actual_view = "None"

        self.frame = None
        self.image = None
        self.classification = None
        self.classifications = None

    def get_home(self, root):
        print("Home View Requested")

        if "Home" == self.actual_view:
            return self.frame

        self.frame = Tk.Frame(root, relief=Tk.FLAT, bd=1, background="green")
        self.frame.grid(row=1, column=0, sticky="nswe")
        return self.frame

    def get_collection(self, root, change_view, classifications):
        print("Collection View Requested")
        if "Collection" == self.actual_view:
            return self.frame

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
            print("Valeur de classification.get_id() ->", classification.get_id())
            bt_image = Tk.Button(fr_fleur, image=classification.get_image(),
                                 command=lambda: change_view("Details", id_classification=classification.get_id()))
            bt_image.grid(row=0, rowspan=2, column=0, padx=5, pady=5)
            # Date
            lb_date_fleur = Tk.Label(fr_fleur, text=classification.get_date(), font=(None, 9))
            lb_date_fleur.grid(row=1, column=1, padx=5, pady=5)

            ligne = ligne + 1

        return self.frame

    def get_details(self, root, classification):
        print("Details View Requested")
        if "Details" == self.actual_view:
            return self.frame

        self.classification = classification

        self.frame = Tk.Frame(root, relief=Tk.FLAT, bd=1, background="red")
        self.frame.grid(row=1, column=0, sticky="nswe")

        cv_image = Tk.Canvas(self.frame, width=root.winfo_screenwidth(),
                             height=root.winfo_screenheight())
        cv_image.pack()

        # Ajout de l'image
        #self.image = Tk.PhotoImage(file='./image/fleur.png')
        cv_image.create_image(0, 0, image=self.classification.get_image(), anchor="nw")

        # Ajout du texte indiquant le type de fleur
        txt_type_de_plante = cv_image.create_text(100, root.winfo_screenheight() * 0.7, anchor=Tk.NW, fill="green")
        cv_image.itemconfig(txt_type_de_plante, text=self.classification.get_type())

        # Ajout du texte indiquant la date de classification
        txt_date_de_classification = cv_image.create_text(100, root.winfo_screenheight() * 0.75, anchor=Tk.NW,
                                                          fill="orange")
        cv_image.itemconfig(txt_date_de_classification, text=self.classification.get_date())

        # Création d'une "Frame" pour contenir la zone de texte contenant la note
        fr_note = Tk.Frame(cv_image, width=int(root.winfo_screenwidth() * 0.20),
                           height=int(root.winfo_screenheight() * 0.10))
        fr_note.place(x=int(root.winfo_screenwidth() * 0.70), y=int(root.winfo_screenheight() * 0.70))
        fr_note.pack_propagate(False)

        # Ajout de la zone de texte dans la "Frame" précédemment créée
        txt_note = Tk.Text(fr_note)
        txt_note.insert(Tk.INSERT, self.classification.get_note())
        txt_note.pack()

        # Ajout du bouton pour supprimer une classification
        bt_importer_photo = Tk.Button(cv_image, text="Supprimer",
                                      bg="#f87e28", fg="#5b5b5b")
        bt_importer_photo.place(x=int(root.winfo_screenwidth() * 0.05), y=int(root.winfo_screenheight() * 0.05))

        return self.frame
