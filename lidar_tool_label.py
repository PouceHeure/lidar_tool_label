import os
import random

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, PolygonSelector
from matplotlib.path import Path

FOLDER_INPUT = "./data/raw"
FOLDER_OUTPUT = "./data/processed"

LABEL_DATA_SELECTED = 1
LABEL_DATA_UNSELECTED = 0

class DataLabel(object): 

    def __init__(self): 
        self._data = []

    def add_data(self,x_line,y_line):
        self._data.append([*x_line,y_line])

    def convert_dataframe(self):
        return pd.DataFrame(self._data)

    def __repr__(self): 
        return "\n".join(self._data)

class Dataset(object): 

    LIMIT_R = 4

    def __init__(self,folder_path):
        self._folder_path = folder_path
        self._files_path = self._load_files_path(folder_path)
        self._current_file_path = None

    def  _load_files_path(self,folder_path): 
        files_path = []
        for file_name in os.listdir(folder_path):
            if os.path.splitext(file_name)[1] == '.csv':
                full_path = os.path.join(folder_path, file_name)
                if os.path.isfile(full_path):
                    files_path.append(full_path)
            else: 
                print(f"can't load this file : {file_name}")
        return files_path 

    def get_files_path(self): 
        return self._files_path

    def get_current_file_path(self): 
        return self._current_file_path

    def next(self,rnd=False,full_data=False):

        if(len(self._files_path) == 0):
            return None 
        file_path = self._files_path[0]
        if(rnd): 
            file_path = random.choice(self._files_path)
        self._current_file_path = file_path
        self._files_path.remove(file_path)

        import csv
        points = []
        with open(self._current_file_path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
            for row in spamreader:
                theta = float(row[0])
                r = float(row[1])
                if(full_data or r < Dataset.LIMIT_R):
                    points.append([theta,r])

        return points 

    def __repr__(self): 
        return "\n".join(self._files_path)

class Controller(object): 

    def __init__(self,dataset,folder_output): 
        self._dataset = dataset
        self._folder_output = folder_output

    def _create_data(self,X,y): 
        if(len(X) != len(y)):
            print("X and y aren't the same dim")
            return 
        # fill data 
        data = DataLabel()
        for i in range(len(X)): 
            data.add_data(X[i],y[i])
        return data 

    def _save_data(self,data): 
        current_path_file = self._dataset.get_current_file_path()
        original_file_name = os.path.basename(current_path_file)
        file_path = os.path.join(self._folder_output,original_file_name)
        data.convert_dataframe().to_csv(file_path,header=False,index=False)

    def create_data(self,X,y): 
        data = self._create_data(X,y)
        self._save_data(data)

class Window(object): 

    def __init__(self,dataset,controller):
        self._controller = controller
        self._dataset = dataset
        self._data = None

        self._fig = plt.figure()
        self._ax = self._fig .add_subplot(111, projection='polar')
        plt.subplots_adjust(bottom=0.2)
        ax_clear = plt.axes([0.7, 0.05, 0.1, 0.075])
        ax_next = plt.axes([0.81, 0.05, 0.1, 0.075])
        self._btn_next = Button(ax_next, 'next')
        self._btn_next.on_clicked(self.on_clicked_next)
        self._btn_clear = Button(ax_clear, 'clear')
        self._btn_clear.on_clicked(self.on_clicked_clear)

        self._alpha_other = 0.01
        self._canvas = self._ax.figure.canvas

        self._refresh_data()

    def on_clicked_next(self, event):
        self._refresh_data()

    def on_clicked_clear(self,event):
        self._plot_data()

    def on_select(self, verts):
        verts = self._polar_to_cart(verts)
        path = Path(verts)
        self._ind = np.nonzero(path.contains_points(self._xys))[0]
        self._fc[:, -1] = self._alpha_other
        self._fc[self._ind, -1] = 1
        self._collection.set_facecolors(self._fc)
        self._canvas.draw_idle()
        X = []
        y = []
        for i in range(len(self._xys)):
            row = self._data[i]
            _y =LABEL_DATA_UNSELECTED
            if(i in self._ind): 
                _y = LABEL_DATA_SELECTED
            X.append([row[0],row[1]])
            y.append(_y)
        self._controller.create_data(X,y)

    def _update_data(self): 
        data = self._dataset.next()
        if(data != None):
            self._data = data
        else: 
            print("all csv has been read")
            exit()

    def _polar_to_cart(self,pts): 
        import math
        new_pts = []
        for point in pts: 
            theta = point[0]
            r = point[1]
            x = math.cos(theta)*r
            y = math.sin(theta)*r
            new_pts.append([x,y])
        return new_pts


    def _plot_data(self): 
        self._plot_clear()
        self._ax.set_title(self._dataset.get_current_file_path())
        thetas = [row[0] for row in self._data]
        rs = [row[1] for row in self._data]
        pts = self._ax.scatter(thetas,rs)

        self._collection = pts
        self._xys = self._polar_to_cart(self._collection.get_offsets())
        self._n_pts = len(self._xys)

        self._fc = self._collection.get_facecolors()
        if len(self._fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self._fc) == 1:
            self._fc = np.tile(self._fc, (self._n_pts, 1))

        self._poly = PolygonSelector(self._ax, self.on_select)
        self._ind = []
        self._canvas.draw()

    def _plot_clear(self):
        self._ax.clear()

    def _refresh_data(self): 
        self._update_data()
        self._plot_data()

    def show(self): 
        plt.show()


if __name__ == "__main__":
    dataset = Dataset(FOLDER_INPUT)
    controller = Controller(dataset,FOLDER_OUTPUT)
    window = Window(dataset,controller)
    window.show()