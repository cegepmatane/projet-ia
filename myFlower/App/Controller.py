import tkinter as Tk
import View
import Model
from PIL import Image, ImageTk
import random
import sqlite3
import io
from datetime import date



class Controller():
    def __init__(self):

        self.view = View.View()
        self.model = Model.Model()
        self.database = Database()

        self.root = Tk.Tk()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=9)
        self.root.grid_columnconfigure(0, weight=3)

        self.fr_header = Tk.Frame(self.root, relief=Tk.FLAT, bd=1, background="orange")
        self.fr_header.grid(row=0,column=0, sticky="nswe")
        self.fr_header.grid_rowconfigure(0, weight=1)
        self.fr_header.grid_columnconfigure(0, weight=1)
        self.fr_header.grid_columnconfigure(1, weight=1)

        self.bt_home = Tk.Button(self.fr_header, text="Accueil", command=lambda : self.change_view("Accueil")
                                 , bg="#5B5B5B", fg="#f87e28")
        self.bt_home.grid(row=0, column=0, sticky="nswe")

        self.bt_collection = Tk.Button(self.fr_header, text="Collection", command=lambda: self.change_view("Collection")
                                 , bg="#5B5B5B", fg="#f87e28")
        self.bt_collection.grid(row=0, column=1, sticky="nswe")

        self.fr_main = self.view.get_home(self.root)

        classification = Model.Classification()
        image_fleur = Image.open('./image/fleur.png')
        largeur_de_base = 100
        pourcent_largeur = (largeur_de_base / float(image_fleur.size[0]))
        hauteur = int((float(image_fleur.size[1]) * float(pourcent_largeur)))
        image_fleur = image_fleur.resize((largeur_de_base, hauteur), Image.ANTIALIAS)
        classification.set_image(image_fleur)
        classification.set_type("Tournesol")
        classification.set_date("" + random.randint(1, 28).__str__() + "/02/2021")
        classification.set_note("Pas de note")
        self.database.save_classification(classification)

        print("Controlleur initialisé")

    def run(self):
        self.root.title("Classification de fleurs")
        self.root.deiconify()
        self.root.mainloop()


    def change_view(self, view=False, id_classification = 0):
        print("Changing view")
        print(id_classification)
        if "Accueil" == view:
            self.fr_main = self.view.get_home(self.root)
            print("Accueil")
        elif "Collection" == view:
            self.fr_main = self.view.get_collection(self.root, self.change_view, self.database.get_collection())
            print("Collection")
        elif "Details" == view:
            self.fr_main = self.view.get_details(self.root, self.database.get_details(id_classification))
            print("Details")
        else:
            print("Error")


class Database(object):
    def __init__(self):
        print("Database initialisée")
        self.collection = None
        self.conn = self.create_connection(self, "database/BDD_Classifications.db")
        self.create_table(self, self.conn, SQL.CREATE_TABLE)
        print(self.conn.cursor().execute("SELECT type, date, note FROM classifications").fetchall())
        self.conn.commit()



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
            parameters = (classification.type, classification.date, classification.note, byteArr)
            c.execute(SQL.INSERT_CLASSIFICATION, parameters)
            self.conn.commit()

        except sqlite3.Error as e:
            print(e)



    def get_collection(self):
        """ Renvoie la totalité des classifications effectuées sur l'appareil sous forme de tableau d'objets
        Classification
        TODO : Passer en BDD"""
        """classifications = []
        image_fleur = Image.open('./image/fleur.png')
        largeur_de_base = 100
        pourcent_largeur = (largeur_de_base / float(image_fleur.size[0]))
        hauteur = int((float(image_fleur.size[1]) * float(pourcent_largeur)))
        image_fleur = image_fleur.resize((largeur_de_base, hauteur), Image.ANTIALIAS)
        image_fleur = ImageTk.PhotoImage(image_fleur)

        for _ in range(100):

            classification = Model.Classification()
            classification.set_image(image_fleur)
            classification.set_type("Tournesol")
            classification.set_date("" + random.randint(1, 28).__str__() + "/02/2021")
            classification.set_note("Pas de note")

            classifications.append(classification)

        self.collection = classifications
        return self.collection"""

        cur = self.conn.cursor()
        results = cur.execute(SQL.SELECT_CLASSIFICATIONS).fetchall()

        classifications = []
        for result in results:
            classification = Model.Classification()
            classification.set_id(result[0])
            classification.set_type(result[1])
            classification.set_date(result[2])
            classification.set_note(result[3])
            classification.set_image(Tk.PhotoImage(data = result[4]))
            classifications.append(classification)

        return classifications

    def get_details(self, id_classification):
        """ Renvoies les détails d'une classification en lui fournissant son id
        TODO : Passer en BDD
        :param id_classification: integer """
        cur = self.conn.cursor()
        parameters = (id_classification,)

        results = cur.execute(SQL.SELECT_CLASSIFICATION, parameters).fetchall()

        print(id_classification)
        for result in results:
            classification = Model.Classification()
            classification.set_id(result[0])
            classification.set_type(result[1])
            classification.set_date(result[2])
            classification.set_note(result[3])
            classification.set_image(Tk.PhotoImage(data=result[4]))

            return classification



class SQL():
    CREATE_TABLE = """CREATE TABLE IF NOT EXISTS classifications (
                                        id integer PRIMARY KEY,
                                        type text NOT NULL,
                                        date text,
                                        note text,
                                        image BLOB
                                    );"""

    INSERT_CLASSIFICATION = """INSERT INTO classifications(type, date, note, image) 
                                VALUES(?, ?, ?, ?)"""

    SELECT_CLASSIFICATIONS = """SELECT * FROM classifications"""

    SELECT_CLASSIFICATION = """SELECT * FROM classifications WHERE id=?"""