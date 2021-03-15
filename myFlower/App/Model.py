import tkinter as Tk


class Model(object):

    def __init__(self):
        print("Modele initialisé")


class Classification(object):

    def __init__(self):
        self.id = "-1"
        self.type = "Type de fleur : Rose"
        self.date = "Analyse faite le : 24/01/2021"
        self.note = "Ceci est une note"
        self.image = Tk.PhotoImage(file='./image/fleur.png')
        self.miniature = Tk.PhotoImage(file='./image/fleur.png')

    def get_id(self):
        return self.id

    def get_image(self):
        return self.image

    def get_miniature(self):
        return self.miniature

    def get_note(self):
        return self.note

    def get_date(self):
        return self.date

    def get_type(self):
        return self.type

    def set_id(self, id):
        self.id = id

    def set_image(self, image):
        self.image = image

    def set_miniature(self, miniature):
        self.miniature = miniature

    def set_note(self, note):
        self.note = note

    def set_date(self, date):
        self.date = date

    def set_type(self, type):
        self.type = type


