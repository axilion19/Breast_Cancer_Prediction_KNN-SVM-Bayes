import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

class App:
    def __init__(self) -> None:
        self.classifier_name = None
        self.Init_Streamlit_Page()
        self.clf = None
        self.df = None
        self.X, self.y = None, None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def Init_Streamlit_Page(self):
        st.title('İleri Seviye Python Modülü')

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Naïve Bayes ')
        )
    def run(self):
        self.get_dataset()
        if self.df is not None:
            self.print_cor_matrix()
            self.generate()
        
    def get_dataset(self):
        uploaded_file = st.sidebar.file_uploader("Bir dosya yükleyin...")

        if uploaded_file is not None:
            st.write(f"## Breast Cancer Wisconsin (Diagnostic) Dataset")
            self.df = pd.read_csv(uploaded_file)
            st.write("\n- Verisetindeki ilk 10 satır. \n")
            st.write(self.df.head(10))
            st.write("\n- Verisetindeki sütunların isimleri. \n")
            st.write(self.df.columns)
            self.data_preprocess()
            self.y = self.df["diagnosis"]
            self.X = self.df.drop("diagnosis", axis=1)

    def data_preprocess(self):
        self.df = self.df.drop(["id", "Unnamed: 32"], axis=1)
        self.df["diagnosis"] = self.df["diagnosis"].map({'M': 1, 'B': 0})
        st.write("\n- Verisetindeki 'id' ve 'Unnamed: 32' sütunları silindi. \n - ‘diagnosis’ sütunundaki M değeri 1, B değeri 0 olacak şekilde değiştirildi. \n - Verisetindeki son 10 satır. \n")
        st.write(self.df.tail(10))

    def print_cor_matrix(self):
        st.write("- Korelasyon matrisi: ")
        fig = sns.lmplot(x = 'radius_mean', y = 'texture_mean', fit_reg = False, scatter_kws={"alpha": 0.3},
            hue = 'diagnosis', palette={1: "red", 0: "green"}, data = self.df)
        st.pyplot(fig)

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            param_grid_svc = {'C': range(1,16)}
            grid_svc = GridSearchCV(SVC(), param_grid_svc , verbose = 3)
            grid_svc.fit(self.X_train, self.y_train)
            self.clf  = SVC(C=grid_svc.best_params_["C"])
        elif self.classifier_name == 'KNN':
            param_grid_knn = {"n_neighbors": range(1,16)}
            grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, verbose=3 )
            grid_knn.fit(self.X_train, self.y_train)
            self.clf  = KNeighborsClassifier(n_neighbors=grid_knn.best_params_["n_neighbors"])
        else:
            param_grid_nb = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
            grid_nb = GridSearchCV(GaussianNB(), param_grid=param_grid_nb)
            grid_nb.fit(self.X_train, self.y_train)
            self.clf  = GaussianNB(var_smoothing=grid_nb.best_params_["var_smoothing"])

    def generate(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)

        self.get_classifier()

        self.clf.fit(self.X_train, self.y_train)
        y_pred = self.clf.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        pre = precision_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        rec = recall_score(self.y_test, y_pred)

        st.write("Accuracy: ", acc)
        st.write("Precision: ", pre)
        st.write("F1_score: ", f1)
        st.write("Recall: ", rec)
        st.write("Confusion Matrix: ")

        conf_matrix = confusion_matrix(self.y_test, y_pred)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [False, True])
        fig, ax = plt.subplots()
        cm_display.plot(ax=ax)
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)