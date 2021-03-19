import tkinter as Tk
from threading import Thread
import cv2

import View
import Model
from PIL import Image, ImageTk
import sqlite3
import io
from datetime import date, datetime
from tkinter.filedialog import askopenfilename

# Imports classification
import os
import tensorflow as tf
import numpy as np
from skimage import transform
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class Controller():
    def __init__(self):

        print(datetime.now().strftime("%Hh%Mm%Ss"),": Initialisation de Controlleur()")
        self.view = View.View()
        self.database = Database()
        self.classificateur = Classificateur()

        self.label = None

        self.root = self.view.get_structure(self.change_view)
        self.actual_view = ""
        self.change_view(view="Accueil")

        self.stream = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        (self.grabbed, self.frame) = self.stream.read()
        self.thread = Thread(target=self.afficher_flux, args=())
        self.thread.start()

        print(datetime.now().strftime("%Hh%Mm%Ss"), ": Controlleur() initialisé.")

    def run(self):
        self.root.title("MyFlower")
        self.root.deiconify()
        self.root.mainloop()

    def change_view(self, view="", id_classification=0):
        print("\nChangement de vue :")
        if "Accueil" == view and self.actual_view != view:
            print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue accueil demandée...")
            self.actual_view = view
            (self.fr_main, self.label) = self.view.get_home(self.root, self.askopenfile,
                                                            self.faire_classification_webcam)
            print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue accueil reçue.")

        elif "Collection" == view and self.actual_view != view:
            print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue collection demandée...")
            self.actual_view = view
            self.fr_main = self.view.get_collection(self.root, self.change_view, self.database.get_collection())
            print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue collection reçue.")

        elif "Details" == view and self.actual_view != view:
            print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue détails demandée...")
            self.actual_view = view
            self.fr_main = self.view.get_details(self.root, self.database.get_details(id_classification),
                                                 self.supprimer_classification, self.modifier_note)
            print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Vue détails reçue.")

        else:
            print("     La vue demandée est la vue actuelle.")

    def askopenfile(self):
        file = askopenfilename(title="Selectionner une image", filetypes=[('Photo à classifier', '*.png')])
        if not file:
            return

        print(file)
        self.classificateur.classifier(Image.open(file), self.change_view, self.database.save_classification)

    def faire_classification_webcam(self):
        self.classificateur.classifier(Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB),"RGB"),
                                       self.change_view, self.database.save_classification)

    def supprimer_classification(self, id):
        self.database.delete_classification(id)
        self.change_view("Collection")

    def modifier_note(self, id, note):
        self.database.update_classification(id, note)

    def afficher_flux(self):
        while 1:
            while self.actual_view == "Accueil":
                (self.grabbed, self.frame) = self.stream.read()
                self.image = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB),"RGB"))
                self.label.configure(image=self.image)
                if cv2.waitKey(1) == ord('q'):
                    break


class Database(object):

    def __init__(self):
        print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Initialisation de Database()")
        self.collection = None
        self.conn = self.create_connection(self, "BDD_Classifications.db")
        self.create_table(self, self.conn, SQL.CREATE_TABLE)
        self.conn.commit()
        print("     ",datetime.now().strftime("%Hh%Mm%Ss"), ": Database() initialisée.")


    @staticmethod
    def create_connection(self, db_file):
        """ Create a database connection to the SQLite database
            specified by db_file
        :param self: self
        :param db_file: database file
        :return: Connection object or None
        """
        conn = None
        try:
            conn = sqlite3.connect(db_file)
            return conn

        except sqlite3.Error as e:
            print(e)

        return conn

    @staticmethod
    def create_table(self, conn, create_table_sql):
        try:
            c = conn.cursor()
            c.execute(create_table_sql)

        except sqlite3.Error as e:
            print(e)

    def save_classification(self, classification):
        try:
            print("Trying to save")
            c = self.conn.cursor()
            byte_io = io.BytesIO()
            classification.image.save(byte_io, format="PNG")
            byteArr = byte_io.getvalue()
            byte_io2 = io.BytesIO()
            classification.miniature.save(byte_io2, format="PNG")
            byteArr2 = byte_io2.getvalue()
            parameters = (classification.type, classification.date, classification.note, byteArr, byteArr2)
            c.execute(SQL.INSERT_CLASSIFICATION, parameters)
            self.conn.commit()
            return self.conn.cursor().execute(SQL.SELECT_LAST_INSERTED_ID).fetchall()[0][0]

        except sqlite3.Error as e:
            print(e)

    def get_collection(self):
        """ Renvoie la totalité des classifications effectuées sur l'appareil sous forme de tableau d'objets
        Classification
        TODO : Passer en BDD"""

        cur = self.conn.cursor()
        results = cur.execute(SQL.SELECT_CLASSIFICATIONS).fetchall()

        classifications = []
        for result in results:
            classification = Model.Classification()
            classification.set_id(result[0])
            classification.set_type(result[1])
            classification.set_date(result[2])
            classification.set_note(result[3])
            classification.set_image(Tk.PhotoImage(data=result[4]))
            classification.set_miniature(Tk.PhotoImage(data=result[5]))
            classifications.append(classification)

        return classifications

    def get_details(self, id_classification):
        """ Renvoies les détails d'une classification en lui fournissant son id
        TODO : Passer en BDD
        :param id_classification: integer """
        cur = self.conn.cursor()
        parameters = (id_classification,)

        results = cur.execute(SQL.SELECT_CLASSIFICATION, parameters).fetchall()

        for result in results:
            classification = Model.Classification()
            classification.set_id(result[0])
            classification.set_type(result[1])
            classification.set_date(result[2])
            classification.set_note(result[3])
            classification.set_image(Tk.PhotoImage(data=result[4]))
            classification.set_miniature(Tk.PhotoImage(data=result[5]))

            return classification

    def delete_classification(self, id_classification):
        cur = self.conn.cursor()
        parameters = (id_classification,)
        cur.execute(SQL.DELETE_CLASSIFICATION_WITH_ID, parameters)
        self.conn.commit()

    def update_classification(self, id_classification, note):
        try:
            c = self.conn.cursor()
            parameters = (note, id_classification)
            c.execute(SQL.UPDATE_NOTE_WITH_ID, parameters)
            print("Update_classification")
            self.conn.commit()
        except sqlite3.Error as e:
            print(e)



class SQL():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS classifications (
                                        id integer PRIMARY KEY,
                                        type text NOT NULL,
                                        date text,
                                        note text,
                                        image BLOB,
                                        miniature BLOB
                                    );"""

    INSERT_CLASSIFICATION = """INSERT INTO classifications(type, date, note, image, miniature) 
                                VALUES(?, ?, ?, ?, ?)"""

    SELECT_CLASSIFICATIONS = """SELECT * FROM classifications"""

    SELECT_CLASSIFICATION = """SELECT * FROM classifications WHERE id=?"""

    SELECT_LAST_INSERTED_ID = """SELECT last_insert_rowid()"""

    DELETE_CLASSIFICATION_WITH_ID = """DELETE FROM classifications WHERE id=?"""

    UPDATE_NOTE_WITH_ID = """UPDATE classifications SET note=? WHERE id=?"""


class Classificateur():

    def __init__(self):
        print("     ",datetime.now().strftime("%Hh%Mm%Ss"),": Initialisation de Classificateur()")
        # Désactiver l'utilisation de GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.model = load_model('myFlower_model.h5')
        ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)
        self.classes = ['Daisy', 'Dandelion', 'Roses', 'Sunflowers', 'Tulipe']
        print("     ",datetime.now().strftime("%Hh%Mm%Ss"), ": Classificateur() initialisé.")


    def classifier(self, image, change_view, save_classification):

        # Charger et formater l'image à classifier
        img = image
        img = np.array(img).astype('float32') / 255
        img = transform.resize(img, (224, 224, 3))
        img = np.expand_dims(img, axis=0)

        # Effectuer la prédiction
        pred_prob = self.model.predict(img)[0]
        pred_class = list(pred_prob).index(max(pred_prob))

        classification = Model.Classification()

        print(max(pred_prob))
        if max(pred_prob) > 0.75:
            classification.set_type(self.classes[pred_class])
        else:
            classification.set_type(self.classes[pred_class] + " (incertain)")

        classification.set_image(image)
        classification.set_note("Ajouter une note")
        classification.set_date(date.today())

        image_fleur = image
        largeur_de_base = 100
        pourcent_largeur = (largeur_de_base / float(image_fleur.size[0]))
        hauteur = int((float(image_fleur.size[1]) * float(pourcent_largeur)))
        image_fleur = image_fleur.resize((largeur_de_base, hauteur), Image.ANTIALIAS)
        classification.set_miniature(image_fleur)

        change_view("Details", id_classification=save_classification(classification))