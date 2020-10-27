import json

MAP_FILE_NAME = 'config/medicine-data-map.json'
with open(MAP_FILE_NAME) as fp:
    jmap = json.load(fp=fp)
    consts = jmap['constants']

SURVIVAL_C = consts['SURVIVAL_C']
CANCER_DIAGNOSIS_C = consts['CANCER_DIAGNOSIS_C']
AGE_AT_DIAGNOSIS_C = consts['AGE_AT_DIAGNOSIS_C']
M_CODE_C = consts['M_CODE_C']
TNM_CODE_C = consts['TNM_CODE_C']
LAST_ORDER_DATE_C = consts['LAST_ORDER_DATE_C']
GENDER_C = consts['GENDER_C']
SMOKING_C = consts['SMOKING_C']
NON_SMOKING_YEAR_C = consts['NON_SMOKING_YEAR_C']
X_SURVIVAL_C = consts['X_SURVIVAL_C']
DELAY_TEST_C = consts['DELAY_TEST_C']
