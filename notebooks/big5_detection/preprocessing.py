# Check wether train and test datasets are distributed similar
# Adverserial Validation
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

def adversarial_validation(X, Y, n_splits=10):
    # Combine both datasets
    sparse_merge = sparse.vstack((X, Y))

    # Label the datasets
    y = np.array([0 for _ in range(X.shape[0])] + [1 for _ in range(Y.shape[0])])

    # Do 10 Fold CV
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    lr_auc = np.array([])
    rf_auc = np.array([])
    for train_idx, test_idx in kfold.split(sparse_merge, y):
        # Run Log Reg
        x_train, y_train = sparse_merge[train_idx], y[train_idx]
        x_test, y_test = sparse_merge[test_idx], y[test_idx]

        log_reg = SGDClassifier(loss='log')
        log_reg.fit(x_train, y_train)
        y_test_prob = log_reg.predict_proba(x_test)[:, 1]
        lr_auc = np.append(lr_auc, roc_auc_score(y_test, y_test_prob))
        # Run RF
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(x_train, y_train)
        y_test_prob = rf.predict_proba(x_test)[:, 1]
        rf_auc = np.append(rf_auc, roc_auc_score(y_test, y_test_prob))

    # Display results
    print('Logisitic Regression AUC : {:.3f}'.format(lr_auc.mean()))
    print('Random Forest AUC : {:.3f}'.format(rf_auc.mean()))

# Inverse Meaning Items have already been recoded before saving into DB
neuro_values = ["neu_item_6","neu_item_21","neu_item_11","neu_item_26","neu_item_41","neu_item_51"]
extra_values = ["ext_item_2","ext_item_7","ext_item_22","ext_item_32","ext_item_37","next_item_52"] # Typo when saving ext_item_52 to db.. but does not matter
off_values = ["off_item_8","off_item_13","off_item_23","off_item_43","off_item_48","off_item_58"]
ver_values = ["ver_item_9", "ver_item_14", "ver_item_24", "ver_item_39","ver_item_49","ver_item_59"]
gew_values = ["gew_item_5","gew_item_10","gew_item_20","gew_item_40","gew_item_50","gew_item_55"]

def Average(l):
    l = list(map(int, l))
    return sum(l) / len(l)


# Scale factor according to the NEO-FFI manual.
def Scale(i):
    i = i * 12
    return i

def compute(dimension, row):
    values = []
    if dimension == "neuro":
        values = [row[i] for i in neuro_values]
    if dimension == "extra":
        values = [row[i] for i in extra_values]
    if dimension == "off":
        values = [row[i] for i in off_values]
    if dimension == "ver":
        values = [row[i] for i in ver_values]
    if dimension == "gew":
        values = [row[i] for i in gew_values]
    mean = Average(values)
    # Scale mean values according to NEO-FFI manual
    # and convert into percentage, with
    # 60 being the max value
    max_value = 60
    skala = (Scale(mean) * 100) / max_value

    return round(skala)