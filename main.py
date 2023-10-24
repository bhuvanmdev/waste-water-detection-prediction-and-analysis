import cv2
import numpy as np
import os
import tkinter as tk
from copy import deepcopy

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import Binarizer
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


class ScrolledFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        tk.Frame.__init__(self, master, **kwargs)

        self.canvas = tk.Canvas(self)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollbar = tk.Scrollbar(self, command=self.canvas.yview)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self.on_canvas_configure)

        self.inner = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

    def on_canvas_configure(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


class tkinter_gui:
    def __init__(self):
        self.image_mode, self.path, self.image_path = None, None, None

    def get_labels1(self):
        return self.image_mode, self.path, self.image_path

    def get_labels2(self):
        return self.model_mode ,*self.li

    def main_loop1(self):
        # Create the main window
        self.root = tk.Tk()
        self.root.title("Water Detection System")
        self.root.geometry(f"{250}x{200}")

        # Create a True/False variable to hold the state of the button
        self.image_mode = tk.BooleanVar()
        self.image_mode.set(False)


        # Create the True/False button-->image dir
        self.true_false_button = tk.Checkbutton(self.root, text="image mode", variable=self.image_mode,
                                           command=self.on_true_false_button_click1)
        self.true_false_button.pack(pady=10)

        #model mode true/false button

        # submit button
        self.submit = tk.Button(self.root, text="Process Input", command=self.on_button_click1)
        self.submit.pack(pady=10)

        # Start the main event loop
        self.root.mainloop()


    def main_loop2(self):
        self.root = tk.Tk()
        self.root.title("Water potability prediction System")
        self.root.geometry(f"{200}x{1700}")
        self.mainroot = self.root

        self.root = ScrolledFrame(self.root)
        self.root.pack(expand=True, fill="both")

        self.model_mode = tk.BooleanVar()
        self.model_mode.set(False)

        self.true_false_button = tk.Checkbutton(self.root.inner, text="model mode", variable=self.model_mode,
                                                command=self.on_true_false_button_click2)
        self.true_false_button.pack(pady=10)

        self.label = tk.Label(self.root.inner, text="DL mode")
        self.label.pack(pady=10)

        tk.Label(self.root.inner, compound='left',text="place").pack(pady=10)
        self.entryp = tk.Entry(self.root.inner, width=30)
        self.entryp.pack(pady=10)

        tk.Label(self.root.inner,text="ph").pack(pady=10)
        self.entrya = tk.Entry(self.root.inner,width=30)
        self.entrya.pack(pady=10)

        tk.Label(self.root.inner, text="hardness").pack(pady=10)
        self.entryb = tk.Entry(self.root.inner, width=30)
        self.entryb.pack(pady=10)

        tk.Label(self.root.inner, text="solids").pack(pady=10)
        self.entryc = tk.Entry(self.root.inner, width=30)
        self.entryc.pack(pady=10)

        tk.Label(self.root.inner, text="chloramites").pack(pady=10)
        self.entryd = tk.Entry(self.root.inner, width=30)
        self.entryd.pack(pady=10)

        tk.Label(self.root.inner, text="sulfates").pack(pady=10)
        self.entrye = tk.Entry(self.root.inner, width=30)
        self.entrye.pack(pady=10)

        tk.Label(self.root.inner, text="conductivity").pack(pady=10)
        self.entryf = tk.Entry(self.root.inner, width=30)
        self.entryf.pack(pady=10)

        tk.Label(self.root.inner, text="Organic_carbon").pack(pady=10)
        self.entryg = tk.Entry(self.root.inner, width=30)
        self.entryg.pack(pady=10)

        tk.Label(self.root.inner, text="trihalomethanes").pack(pady=10)
        self.entryh = tk.Entry(self.root.inner, width=30)
        self.entryh.pack(pady=10)

        tk.Label(self.root.inner, text="turbudity").pack(pady=10)
        self.entryi = tk.Entry(self.root.inner, width=30)
        self.entryi.pack(pady=10)

        self.submit = tk.Button(self.root.inner, text="Process Input", command=self.on_button_click2)
        self.submit.pack(pady=10)

        self.li = [self.entryp,self.entrya, self.entryb, self.entryc, self.entryd, self.entrye, self.entryf, self.entryg,
                   self.entryh, self.entryi]

        # Start the main event loop
        self.root.inner.mainloop()


    def main_loop3(self):
        self.root = tk.Tk()
        self.root.title("Water analysis System")
        self.root.geometry(f"{200}x{250}")

        self.graph = tk.BooleanVar()
        self.graph.set(False)

        self.label = tk.Label(self.root,text="potability analysis")
        self.label.pack(pady=10)

        # Create the True/False button
        self.true_false_button = tk.Checkbutton(self.root, text="analysis type", variable=self.graph,
                                                command=self.on_true_false_button_click3)
        self.true_false_button.pack(pady=10)


        # submit button
        self.submit = tk.Button(self.root, text="Process Input", command=self.on_button_click3)
        self.submit.pack(pady=10)
        self.root.mainloop()

    def on_button_click1(self):
        if not self.image_mode.get():
            self.image_mode = False
        else:
            self.image_mode = True
            self.image_path = self.entry1.get()
            self.path = os.listdir(self.image_path)
        self.root.destroy()

    def on_button_click2(self):
        for x in range(10):
            if self.li[x].get() in (None,''):
                print(self.li[x].get())
                self.li[x] = 0
            else:
                self.li[x] = self.li[x].get()
        self.root.destroy()
        self.mainroot.destroy()

    def on_button_click3(self):
        if self.graph.get():
            self.entry3 = self.entry3.get()
        self.graph = self.graph.get()
        self.root.destroy()

    def on_true_false_button_click1(self):
        if self.image_mode.get():
            # Create the input bar (Entry widget)
            self.input_label1 = tk.Label(self.root, text="input images directory")
            self.input_label1.pack(pady=5)
            self.entry1 = tk.Entry(self.root, width=30)
            self.entry1.pack(pady=10)
            # Create a button to process the input value
        else:
            self.entry1.destroy()
            self.input_label1.destroy()

    def on_true_false_button_click2(self):
        if self.model_mode.get():
            self.label.config(text="ML mode")
        else:
            self.label.config(text="DL mode")

    def on_true_false_button_click3(self):
        if self.graph.get():
            # Create the input bar (Entry widget)
            self.label.config(text="impurity analysis")
            self.input_label = tk.Label(self.root, text="impurity to compare")
            self.input_label.pack(pady=5)
            self.entry3 = tk.Entry(self.root, width=30)
            self.entry3.pack(pady=10)
            # Create a button to process the input value
        else:
            self.label.config(text="potability analysis")
            self.entry3.destroy()
            self.input_label.destroy()


class water_detection_prediction_analysis:
    def __init__(self):
        self.model_init_ml()
        self.model_init_dl()
        self.database_init()
        self.tkinter_object = tkinter_gui()

    def model_init_ml(self):
        # import dataset
        df = pd.read_csv("train_dataset.csv")

        columns = [i for i in df.columns]

        # the input and output data is created drom the dataframe
        data = df.drop('Potability', axis=1)
        answer = df['Potability']

        # training and testing data is created from the input and output data
        x_train, x_test, y_train, y_test = train_test_split(data, answer, test_size=0.25, random_state=42)
        self.tree = RandomForestClassifier(criterion='entropy', max_depth=30, max_features='log2', min_samples_leaf=4,
                                      min_samples_split=4, n_estimators=110)

        # data given to the supervised learning model
        self.tree.fit(x_train, y_train)

        # prediction obtained from testing data
        prediction = self.tree.predict(x_test)
        c = 0

        # the prediction is tested with the actual output to check its accuracy
        for x, y in zip(prediction, y_test):
            if x == y: c += 1
        print(f"{c}/{len(y_test)} and {int(c / len(y_test) * 100)}")

        # we get accuracy_score and confusion matrix for analysis of our tuned model
        # print(accuracy_score(y_test, prediction), confusion_matrix(y_test, prediction, normalize='true'), sep='\n')
        print('ML MODEL TRAINED!!!\n')

    def model_init_dl(self):
        self.model_dl = tf.keras.models.load_model('water_model.hdf5')
        self.binarizer = Binarizer(threshold=0.5)
        print('DL model loaded!!!\n')


    def database_init(self):
        uri = open('uri.txt').readlines()[0]

        # Create a new client and connect to the server
        self.client = MongoClient(uri, server_api=ServerApi('1'))

        # Send a ping to confirm a successful connection
        try:
            self.client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            self.coll = self.client['water_sample_data']['sample_collection'] # put your own path
        except Exception as e:
            print(e)

        cursor = self.coll.find({})

        # Convert cursor to a list of dictionaries
        self.data_list = list(cursor)
        print(f'DATA RETRIEVING IS DONE OF {len(self.data_list)}!!!\n')

    def detection(self):
        self.tkinter_object.main_loop1()
        image_mode, path, image_path = self.tkinter_object.get_labels1()
        # if GUI is abruptly closed default is False
        if type(image_mode) is not bool:
            image_mode = False

        # activate camera
        if image_mode is False:
            cap = cv2.VideoCapture(0)

        while True:
            if not image_mode:
                ret, image = cap.read()
                if not ret or cv2.waitKey(10) == 27: break
            else:
                if path == []: break
                fil = path.pop(0)
                image = cv2.imread(image_path + "\\" + fil)

            image_real = deepcopy(image)
            # Convert to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define lower and upper thresholds for water color (example: black,brown, dark colors)

            lower_brown = np.array([0, 50, 50])  # Lower threshold for brown color10,20
            upper_brown = np.array([50, 255, 255])

            lower_dark = np.array([10, 50, 30])
            upper_dark = np.array([30, 255, 100])

            lower_black = np.array([0, 0, 0])
            upper_black = np.array([179, 100, 50])

            # Create a mask using inRange() to isolate water color

            # brown mask
            maskb = cv2.inRange(hsv_image, lower_brown, upper_brown)

            # dark mask
            maskdb = cv2.inRange(hsv_image, lower_dark, upper_dark)

            # black mask
            maskbl = cv2.inRange(hsv_image, lower_black, upper_black)

            # final mask
            mask = cv2.bitwise_or(maskb, cv2.bitwise_or(maskbl, maskdb))  # white=1,black=0

            # Apply morphological operations-->cleaning
            kernel = np.ones((5, 5), np.uint8)

            # erode and dilate the masked image
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Find contours of water regions
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # take the biggest contour and rectangulate around it
            if len(contours) != 0:
                contour = max(contours, key=len)
                # cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)#draws to the original given image itself
                x, y, w, h = cv2.boundingRect(contour)
                if w * h > 1000:
                    image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Show the results
            cv2.imshow("Detected Image", image)
            cv2.imshow("Original Image",image_real)
            if image_mode:
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        cv2.waitKey(0)
        if not image_mode: cap.release()
        print("DETECTION DONE SUCCESSFULLY!!!\n")
        cv2.destroyAllWindows()


    def prediction(self):
        root = tk.Tk()
        root.geometry(f"{200}x{250}")
        tk.Label(root,text="Total_predictions").pack(pady=10)
        n = tk.Entry(root,width=30)
        n.pack(pady=5)
        tk.Button(root, text="Process Input", command=(lambda: [setattr(root, "result", n.get()),root.destroy()])).pack(pady=10)
        root.mainloop()
        for _ in range(int(root.result)):
            self.tkinter_object.main_loop2()
            model_mode,place,ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity = self.tkinter_object.get_labels2()
            data = {
                "city": place,
                "ph": ph,
                "Hardness": Hardness,
                "Solids": Solids,
                "Chloramines": Chloramines,
                "Sulfate": Sulfate,
                "Conductivity": Conductivity,
                "Organic_carbon": Organic_carbon,
                "Trihalomethanes": Trihalomethanes,
                "Turbidity": Turbidity
            }

            print("data uploaded successfully")
            # predicts the drinkability from user input data
            def predict_ml(l):
                out = self.tree.predict([l])#[a, b, c, d, e, f, g, h, i]
                return out

            def predict_dl(l):
                X = np.array(l, dtype=np.float32)
                X = X.reshape(1, -1)
                output_X = self.binarizer.transform(X)
                return self.binarizer.transform(self.model_dl.predict(X))

            #tkinter_model
            result = {0:predict_dl,1:predict_ml}[model_mode.get()]([ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon,
                                        Trihalomethanes, Turbidity])

            #self.coll.insert_one(data|{'Potability':str(int(result))})

            if result == 0:
                print(f"\n{_}'s data provided water is not potable")
            else:
                print(f"\n{_}'s data provided water is potable")
        print(f"{root.result} PREDICTION(s) DONE SUCCESSFULLY!!!\n")
        self.client.close()

    def analysis(self):
        self.tkinter_object.main_loop3()
        graph = self.tkinter_object.graph
        cities = [data["city"] for data in self.data_list]
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 6))
        if graph:
            impurity = self.tkinter_object.entry3
            poll_levels = [round(float(data[impurity]),2) for data in
                               self.data_list]  # Assuming "Chloramines" is the key for chloride levels

            # Create a DataFrame for visualization
            df1 = pd.DataFrame({"City": cities, f"{impurity} Levels": poll_levels})

            # Create a box plot using Seaborn

            sns.boxplot(x="City", y=f"{impurity} Levels", data=df1)
            plt.title(f"{impurity} Levels by City")
            plt.xlabel("City")
            plt.ylabel(f"{impurity} Levels")
        else:
            potabilities = [data["Potability"] for data in self.data_list]

            # Create a DataFrame for visualization
            df = pd.DataFrame({"City": cities, "Potability": potabilities})

            # Count the occurrences of each combination of city and potability
            count_data = df.groupby(["City", "Potability"]).size().reset_index(name="Count")

            # Create a bar plot using Seaborn
            sns.barplot(x="City", y="Count", hue="Potability", data=count_data)
            plt.title("Potability Distribution by City")
            plt.xlabel("City")
            plt.ylabel("Count")
        plt.xticks(rotation=90)
        plt.show()
        print("ANALYSIS DONE SUCCESSFULLY!!!\n")

if  __name__ == '__main__':
    new1 = water_detection_prediction_analysis()
    new1.detection()
    new1.prediction()
    new1.analysis()
