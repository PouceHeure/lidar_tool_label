import os
import random

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, PolygonSelector
from matplotlib.path import Path

FOLDER_INPUT = "./data/raw"
FOLDER_OUTPUT = "./data/processed"

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

    def next(self,rnd=False):
        if(len(self._files_path) == 0):
            return pd.DataFrame()
        file_path = self._files_path[0]
        if(rnd): 
            file_path = random.choice(self._files_path)
        self._current_file_path = file_path
        self._files_path.remove(file_path)
        return pd.read_csv(file_path,header=None)

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

        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)
        axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(axnext, 'next')
        self.bnext.on_clicked(self.on_clicked_next)
        self.baext = Button(axprev, 'clear')
        self.baext.on_clicked(self.on_clicked_clear)

        self.alpha_other = 0.10
        self.canvas = self.ax.figure.canvas

        self.refresh_data()

    def on_clicked_next(self, event):
        self.refresh_data()

    def on_clicked_clear(self,event):
        self.plot_data()

    def update_data(self): 
        data = self._dataset.next()
        if(not data.empty):
            self._data = data

    def plot_data(self): 
        self.plot_clear()
        self.ax.set_title(self._dataset.get_current_file_path())
        pts = self.ax.scatter(self._data.iloc[:,0],self._data.iloc[:,1])

        self.collection = pts
        self.xys = self.collection.get_offsets()
        self.Npts = len(self.xys)

        self.fc = self.collection.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.poly = PolygonSelector(self.ax, self.onselect)
        self.ind = []
        self.canvas.draw()

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_other
        self.fc[self.ind, -1] = 1
        self.collection.set_facecolors(self.fc)
        self.canvas.draw_idle()

        X = []
        y = []
        for row in self._data.itertuples():
            _y = "unselect"
            if(row[0] in self.ind): 
                _y = "select"
            X.append([row[1],row[2]])
            y.append(_y)
        self._controller.create_data(X,y)

    def plot_clear(self):
        self.ax.clear()

    def refresh_data(self): 
        self.update_data()
        self.plot_data()

    def show(self): 
        plt.show()


if __name__ == "__main__":
    dataset = Dataset(FOLDER_INPUT)
    controller = Controller(dataset,FOLDER_OUTPUT)
    window = Window(dataset,controller)
    window.show()