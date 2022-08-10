import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import pandas as pd
import tkinter as tk
from scipy.spatial.transform import Rotation as R

LARGE_FONT= ("Verdana", 12)

class MergeDatasetsGui(tk.Tk):

    def __init__(self, df_1, df_2, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "Merge datasets")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        self.df_1 = df_1 # =px4
        self.df_2 = df_2 # =opti
        self.start_time_offset = pd.Timedelta('0L')
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        

        frame = StartPage(container, self)

        self.frames[StartPage] = frame

        frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()

        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        label = tk.Label(self, text="Merge", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        self.controller = controller

        self.df_1 = controller.df_1
        self.df_2 = controller.df_2
        self.df_2.loc[:, 'pitch'][self.df_2.loc[:, 'pitch'] < -0.5*np.pi] = self.df_2.loc[:, 'pitch'] + np.pi
        self.df_2.loc[:, 'pitch'][self.df_2.loc[:, 'pitch'] > 0.5*np.pi] = self.df_2.loc[:, 'pitch'] - np.pi
        # self.df_2.loc[:, 'roll'][self.df_2.loc[:, 'roll'] < -0.5*np.pi] = self.df_2.loc[:, 'roll'] + np.pi
        # self.df_2.loc[:, 'roll'][self.df_2.loc[:, 'roll'] > 0.5*np.pi] = self.df_2.loc[:, 'roll'] - np.pi
        self.df_2.loc[:, 'yaw'][self.df_2.loc[:, 'yaw'] < -0.5*np.pi] = self.df_2.loc[:, 'yaw'] + np.pi
        self.df_2.loc[:, 'yaw'][self.df_2.loc[:, 'yaw'] > 0.5*np.pi] = self.df_2.loc[:, 'yaw'] - np.pi
        self.df_2_copy = self.df_2.copy()

        slider = tk.Scale(self,
                            from_=-15000,
                            to=15000,
                            resolution=1,
                            command=self.update_offset,
                            orient=tk.HORIZONTAL,
                            length=400)
        slider.pack(side=tk.BOTTOM)

        rotation_slider_x = tk.Scale(self,
                            from_=-np.pi,
                            to=np.pi,
                            resolution=0.01,
                            command=self.update_rotation_x,
                            orient=tk.HORIZONTAL,
                            length=400, 
                            label='rotation around x')

        rotation_slider_y = tk.Scale(self,
                            from_=-np.pi,
                            to=np.pi,
                            resolution=0.01,
                            command=self.update_rotation_y,
                            orient=tk.HORIZONTAL,
                            length=400, 
                            label='rotation around y')

        rotation_slider_z = tk.Scale(self,
                            from_=-np.pi,
                            to=np.pi,
                            resolution=0.01,
                            command=self.update_rotation_z,
                            orient=tk.HORIZONTAL,
                            length=400, 
                            label='rotation around z')
        rotation_slider_x.pack(side=tk.BOTTOM)
        rotation_slider_y.pack(side=tk.BOTTOM)
        rotation_slider_z.pack(side=tk.BOTTOM)

        close_button = tk.Button(self, 
                                    command=self.controller.destroy,
                                    text="Merge datasets with this start value")
        close_button.pack(side=tk.BOTTOM)

        f = Figure(figsize=(20,10), dpi=100)
        self.pitch = f.add_subplot(311)
        self.roll = f.add_subplot(312)
        self.yaw = f.add_subplot(313)

        self.canvas = FigureCanvasTkAgg(f, self)
        self.redraw_plot()

        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_offset(self, event):
        self.controller.start_time_offset = pd.Timedelta(f'{int(event)}L')
        self.redraw_plot()

    def update_rotation_x(self, rotation):
        self.controller.rotation_x = float(rotation)
        # rot_mat = R.from_euler('x', rotation).as_matrix()
        # self.df_2.loc[:, ['pitch', 'roll', 'yaw']] =  np.transpose(rot_mat @ self.df_2_copy.loc[:, ['pitch', 'roll', 'yaw']].to_numpy().T)
        self.redraw_plot()

    def update_rotation_y(self, rotation):
        self.controller.rotation_y = float(rotation)
        # rot_mat = R.from_euler('y', rotation).as_matrix()
        # self.df_2.loc[:, ['pitch', 'roll', 'yaw']] =  np.transpose(rot_mat @ self.df_2_copy.loc[:, ['pitch', 'roll', 'yaw']].to_numpy().T)
        self.redraw_plot()

    def update_rotation_z(self, rotation):
        self.controller.rotation_z = float(rotation)
        # rot_mat = R.from_euler('z', rotation).as_matrix()
        # self.df_2.loc[:, ['pitch', 'roll', 'yaw']] =  np.transpose(rot_mat @ self.df_2_copy.loc[:, ['pitch', 'roll', 'yaw']].to_numpy().T)
        self.redraw_plot()


    def redraw_plot(self):
        self.pitch.clear()
        self.roll.clear()
        self.yaw.clear()

        start_val = self.controller.start_time_offset

        rot_mat = R.from_euler('xyz', [self.controller.rotation_x, self.controller.rotation_y, self.controller.rotation_z]).as_matrix()
        # rot_quat = R.from_euler('xyz', [self.controller.rotation_x, self.controller.rotation_y, self.controller.rotation_z]).as_quat()
        self.df_2.loc[:, ['pitch', 'roll', 'yaw']] =  np.transpose(rot_mat @ self.df_2_copy.loc[:, ['pitch', 'roll', 'yaw']].to_numpy().T)
        # self.df_2.loc[:, ['quat_x', 'quat_y', 'quat_z', 'quat_w']] = rot_quat 
        

        # self.pitch.plot(np.linspace(start_val, start_val + len(self.df_1), len(self.df_2)), self.df_2.loc[:, 'pitch'])
        self.pitch.plot(self.df_2.loc[:, 'Time'] +  start_val, self.df_2.loc[:, 'pitch'])
        self.pitch.plot(self.df_1.loc[:, 'Time'], self.df_1.loc[:, 'pitch_kalman'])
        self.pitch.set_ylim([-0.3, 0.3])
        self.roll.plot(self.df_2.loc[:, 'Time']  +  start_val, self.df_2.loc[:, 'roll'])
        self.roll.plot(self.df_1.loc[:, 'Time'], self.df_1.loc[:, 'roll_kalman'])
        # self.roll.set_ylim([-1, 1])
        self.yaw.plot(self.df_2.loc[:, 'Time']  +  start_val, self.df_2.loc[:, 'yaw'])
        self.yaw.plot(self.df_1.loc[:, 'Time'], self.df_1.loc[:, 'yaw_kalman'])
        self.yaw.set_ylim([-0.3, 0.3])

        self.canvas.draw()
        