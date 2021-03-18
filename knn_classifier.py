import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score , classification_report
import time
import datetime
import pandas as pd
import sys


image_dim = (40, 40)


def chiSquared(p,q):
    chi =0.5*np.sum((p-q)**2/(p+q+1e-6))
    return chi


def load_images(PATH, folder):
    # load and preprocess images
    X = []
    y = []
    for i in range(27):
        path = '' + PATH + folder + '/' + str(i) + '/'

        for file in os.listdir(path):
            image_to_open = '' + path + file
            loaded_image = cv2.imread(image_to_open)
            loaded_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
            shape = loaded_image.shape
            final_size = shape[0] if shape[0] > shape[1] else shape[1]
            new_width = int((final_size - shape[1]) / 2)
            new_height = int((final_size - shape[0]) / 2)
            processed_image = cv2.copyMakeBorder(loaded_image,
                                                 top=int(shape[0] + new_height),
                                                 bottom=int(shape[0] + new_height),
                                                 left=int(shape[1] + new_width),
                                                 right=int(shape[1] + new_width),
                                                 borderType=cv2.BORDER_CONSTANT,
                                                 value=0
                                                 )
            processed_image = cv2.resize(loaded_image, image_dim)
            X.append(processed_image)
            y.append(i)
            # if i==2:
            #     plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            #     plt.show()


    return X, y


def feature_extraction(dataset):
    # Extracting features
    dataset_features = []
    for img in dataset:
        ch_hog = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=False,
                         block_norm="L2")
        dataset_features.append(ch_hog)
    return dataset_features


def create_and_train_model(X, y, metric, k):
    # creating model
    classifier = KNeighborsClassifier(n_neighbors=k, weights='distance', metric=metric)
    # training model
    classifier.fit(X, y)
    return classifier


def create_output_files(y_true, y_pred, model):
    print("Creating output files...")
    accuracy = []
    ac = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    # print(ac)
    try:
        for i in range(27):
            if i>9:
                tmp = "" + str(i) + "        " + str(ac[str(i)]['precision'])
            else:
                tmp = "" + str(i) + "         " + str(ac[str(i)]['precision'])
            accuracy.append(tmp)
    except Exception as e:
        print(e)
        
    # print(accuracy)
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm).to_csv("./confusion_matrix.csv")

    try:
        txtfile = open("./results.txt", "w")
        txtfile.write("k = " + str(model.n_neighbors) + ", ")
        txtfile.write("distance algorithm = ")
        txtfile.write("euclidean" if str(model.metric) == "minkowski" else "chi square")
        txtfile.write("\n\n")
        txtfile.write("Letter    Accuracy \n")
        for a in accuracy:
            txtfile.write(a + "\n")
        txtfile.close()

    except IOError as e:
        print(e)


def get_best_model(train, train_labels):

    # split train images
    X_train, X_test, y_train, y_test = train_test_split(train, train_labels, test_size=0.1, random_state=42)

    # get best euclidean
    print("finding best model...")

    max_accuracy_euclidean = 0
    k_best_euclidean = 0
    max_accuracy_chi = 0
    k_best_chi = 0

    for i in range(1, 16, 2):
        print("testing accuracy for k={}".format(i))

        classifierE = create_and_train_model(X_train, y_train, 'minkowski', i)
        classifierC = create_and_train_model(X_train, y_train, chiSquared, i)

        y_predE = classifierE.predict(X_test)
        accuracyE = accuracy_score(y_test, y_predE)
        if accuracyE > max_accuracy_euclidean:
            max_accuracy_euclidean = accuracyE
            k_best_euclidean = i

        y_predC = classifierC.predict(X_test)
        accuracyC = accuracy_score(y_test, y_predC)
        if accuracyC > max_accuracy_chi:
            max_accuracy_chi = accuracyC
            k_best_chi = i

        # print('the accuracy for Euclidean: {} \nthe  accuracy for Chi-Square: {}\nn_neighbors={}'.format(
        #     max_accuracy_euclidean, max_accuracy_chi, i))


    print('the best accuracy for Euclidean: {}\nn_neighbors={} \nthe best accuracy for Chi-Square: {}\nn_neighbors={}'.format(max_accuracy_euclidean, k_best_euclidean,  max_accuracy_chi, k_best_chi))
    if max_accuracy_chi > max_accuracy_euclidean:
        return chiSquared, k_best_chi
    else:
        return 'minkowski', k_best_euclidean


def engine(PATH):

    # print current time and date
    start_time = time.time()
    print("current date and time: " + str(datetime.datetime.now()))

    # load and preprocess train images
    X_train, y_train = load_images(PATH, 'TRAIN')

    # # extract features
    X_train = feature_extraction(X_train)

    # get the best model
    metric, k = get_best_model(X_train, y_train)

    # load and preprocess test images
    X_test, y_test = load_images(PATH, 'TEST')

    # extract features
    X_test = feature_extraction(X_test)

    # create and train the best model found
    print("Creating best model")
    model = create_and_train_model(X_train, y_train, metric, k)

    # prediction for X_test
    print("Testing model")
    y_pred = model.predict(X_test)

    # create csv and txt output files
    create_output_files(y_test, y_pred, model)

    # print program run time
    print("--- {} minutes ---".format(round((time.time() - start_time)/60, 3)))


if __name__ == '__main__':
    PATH = sys.argv[1]
    PATH += '/'
    engine(PATH)


