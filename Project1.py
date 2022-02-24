from sklearn.utils import shuffle
import numpy as np
import csv
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Length of data + label
dataLabel = 20;
# Length of data without label (Feature Vector)
dataLength = 19

# .csv files created in matlab
with open('label_0.csv') as csvfile:
    reader0 = csv.reader(csvfile, delimiter=',')
    c0 = np.zeros((1000,dataLabel ))
    i = 0
    for row0 in reader0:
        c0[i, :] = row0
        i += 1

with open('label_1.csv') as csvfile:
    reader1 = csv.reader(csvfile, delimiter=',')
    c1 = np.zeros((1000,dataLabel))
    i=0
    for row1 in reader1:
        c1[i,:] = row1
        i += 1

# Concatenate the two data sets
data = np.concatenate((c0 , c1), axis=0)
# Features are the first 19 sets of data
features = data[:,0:dataLength]
# Label is 0 or 1 appended to the end of the data
label = data[:,dataLength]
# Shuffle
features, label = shuffle(features, label)
# Use 5-fold cross-validation to train a logistic regression classifier.
# Assign 1600 features for training and 400 features for testing (20%)
kf = KFold(n_splits=5)
kf.get_n_splits(features)
#Logistic Regression
j = 0
AreaUnderCurve = np.zeros(5)
test_accuracy = np.zeros(5)
# Compute the sensitivity, specificity and accuracy
for train_index, test_index in kf.split(features):
    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    features_train, features_test = features[train_index], features[test_index]
    label_train, label_test = label[train_index], label[test_index]

    trained_model = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(features_train,label_train)
    predict = trained_model.predict(features_test)
    predict_probability = trained_model.predict_proba(features_test)[::, 1]
    scorei = accuracy_score(label_test, predict)
    test_accuracy[j] = scorei
    fpr, tpr, th = metrics.roc_curve(label_test, predict_probability)
    auc = metrics.roc_auc_score(label_test, predict)
    AreaUnderCurve[j] = auc

    conf = confusion_matrix(label_test, predict)
    sensitivity = conf[0, 0] / (conf[0, 0] + conf[1, 0])
    specificity = conf[1, 1] / (conf[0, 1] + conf[1, 1])
    print(specificity, "Specificity: %0.2f" % np.mean(specificity))
    print(sensitivity, "Sensitivity: %0.2f" % np.mean(sensitivity))

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='(AUC = %0.2f) ROC fold %d ' % (auc, (j+1)))
    #print(th)
    j+=1
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Logistic Regression')
plt.legend(loc="lower right")
plt.show()

print(test_accuracy, "Test Accuracy: %0.2f"% np.mean(test_accuracy))
print(AreaUnderCurve, "AUC: %0.2f"% np.mean(AreaUnderCurve))

# Train another classifier of your choice (e.g. LDA, SVM,
# neural network). Plot the ROC curve and compute the AUC.
# LDA
j = 0
AreaUnderCurve = np.zeros(5)
test_accuracy = np.zeros(5)
for train_index, test_index in kf.split(features):
    print("TRAIN:", train_index.shape, "TEST:", test_index.shape)
    features_train, features_test = features[train_index], features[test_index]
    label_train, label_test = label[train_index], label[test_index]

    trained_model = LDA().fit(features_train, label_train)
    predict = trained_model.predict(features_test)
    predict_probability = trained_model.predict_proba(features_test)[::, 1]
    scorei = accuracy_score(label_test, predict)
    test_accuracy[j] = scorei
    fpr, tpr, _ = metrics.roc_curve(label_test, predict_probability)
    auc = metrics.roc_auc_score(label_test, predict)
    AreaUnderCurve[j] = auc
    conf = confusion_matrix(label_test, predict)
    sensitivity = conf[0, 0] / (conf[0, 0] + conf[1, 0])
    specificity = conf[1, 1] / (conf[0, 1] + conf[1, 1])
    print(specificity, "Specificity: %0.2f" % np.mean(specificity))
    print(sensitivity, "Sensitivity: %0.2f" % np.mean(sensitivity))

    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='(AUC = %0.2f) ROC fold %d ' % (auc, (j+1)))

    j+=1

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for LDA')
plt.legend(loc="lower right")
plt.show()

print(test_accuracy, "Test Accuracy: %0.2f"% np.mean(test_accuracy))
print(AreaUnderCurve, "Area Under Curve: %0.2f"% np.mean(AreaUnderCurve))



