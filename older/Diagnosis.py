import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


def encode_diagnosis(diagnosis):
    le = LabelEncoder()
    le.fit(['CTRL', 'MSA', 'PD', 'PSP'])
    # This is the encoding:
    # CTRL -----> 0 -----> [1,0,0,0]
    # MSA ----->  1 -----> [0,1,0,0]
    # PD -----> 2 -----> [0,0,1,0]
    # PSP -----> 3 -----> [0,0,0,1]

    diagnosis = le.transform([diagnosis])
    oneHotDiagnosis = np.zeros((1, 4), dtype='uint8')
    idx = diagnosis[0]
    oneHotDiagnosis[0][idx] = 1
    return oneHotDiagnosis[0]


def decode_diagnosis(diagnosis):
    # This is the encoding:
    # CTRL -----> 0 -----> [1,0,0,0]
    # MSA ----->  1 -----> [0,1,0,0]
    # PD -----> 2 -----> [0,0,1,0]
    # PSP -----> 3 -----> [0,0,0,1]
    if diagnosis[0] == 1: return 'CTRL'
    if diagnosis[1] == 1: return 'MSA'
    if diagnosis[2] == 1: return 'PD'
    if diagnosis[3] == 1: return 'PSP'
    return ''


def show_class_distribution(diagnoses, test_type, plot=1):
    CTRL = sum(1 for i, diagnose in enumerate(diagnoses) if decode_diagnosis(diagnose) == 'CTRL')
    PD = sum(1 for i, diagnose in enumerate(diagnoses) if decode_diagnosis(diagnose) == 'PD')
    PSP = sum(1 for i, diagnose in enumerate(diagnoses) if decode_diagnosis(diagnose) == 'PSP')
    MSA = sum(1 for i, diagnose in enumerate(diagnoses) if decode_diagnosis(diagnose) == 'MSA')

    # print
    print('There are {} CTRL subjects, {} MSA, {} PD and {} PSP'.format(CTRL, MSA, PD, PSP))

    # plot
    if plot == 1:
        plt.bar(['CTRL', 'MSA', 'PD', 'PSP'], [CTRL, MSA, PD, PSP])
        plt.title('Number of signals recorded by diagnosis {}'.format(test_type))
        plt.show()

    return


def equals(diag1, diag2):
    return diag1 == diag2


def get_diagnosis_names():
    d = ['CTRL', 'MSA', 'PD', 'PSP']
    return d


def get_diagnosis_names_plot():
    d = {0: 'HC',
         1: 'MSA',
         2: 'PD',
         3: 'PSP'}
    return d


def get_diagnosis_number():
    return 4
