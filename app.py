import streamlit as st
import joblib
import pandas as pd
import shap
import plotly.figure_factory as ff
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score




model = joblib.load("models/svm_model.pkl")
columns = joblib.load("models/columns.pkl")
background = joblib.load("models/background.pkl").values




st.set_page_config(page_title="Heart Disease App", layout="wide")



st.title("Heart Disease Prediction System")


# Data entry:

age = st.number_input("Age")
cp = st.selectbox("Chest Pain Type", [0,1,2,3,4])
bp = st.number_input("BP")
chol = st.number_input("Cholesterol")
fbs = st.selectbox("FBS over 120", [0,1])
ekg = st.selectbox("EKG results", [0,1,2])
max_hr = st.number_input("Max HR")
angina = st.selectbox("Exercise Angina", [0,1])
st_depression = st.number_input("ST Depression")
slope = st.selectbox("Slope of ST", [0,1,2,3])
vessels = st.number_input("Number of vessels fluro")
thallium = st.selectbox("Thallium", [0,1,2,3,4,5,6,7])

gender = st.selectbox("Gender", ["Female", "Male"])
gender_female = 1 if gender == "Female" else 0
gender_male = 1 if gender == "Male" else 0

work = st.selectbox("Work Type",
                    ["Govt_job","Never_worked","Private","Self-employed"])
work_gov = 1 if work == "Govt_job" else 0
work_never = 1 if work == "Never_worked" else 0
work_private = 1 if work == "Private" else 0
work_self = 1 if work == "Self-employed" else 0

smoke = st.selectbox("Smoking Status",
                     ["Unknown","formerly smoked","never smoked","smokes"])
smoke_unknown = 1 if smoke == "Unknown" else 0
smoke_former = 1 if smoke == "formerly smoked" else 0
smoke_never = 1 if smoke == "never smoked" else 0
smoke_smokes = 1 if smoke == "smokes" else 0




input_data = pd.DataFrame([[age, cp, bp, chol, fbs, ekg, max_hr,
                            angina, st_depression, slope, vessels, thallium,
                            gender_female, gender_male,
                            work_gov, work_never, work_private, work_self,
                            smoke_unknown, smoke_former, smoke_never, smoke_smokes]],
                          columns=columns)



# Prediction

if st.button("Predict"):

    pred = model.predict(input_data)

    st.subheader("Prediction Result")

    if pred[0] == 1:
        st.error("Heart Disease Detected")
    else:
        st.success("No Heart Disease")




# SHAP Explanation

if st.button("Explain Prediction"):

    explainer = shap.KernelExplainer(
    lambda x: model.predict_proba(x)[:,1],
    background
)

    shap_values = explainer.shap_values(input_data.values)

    shap_df = pd.DataFrame({
    "Feature": columns,
    "Impact": shap_values[0]
})



    shap_df = shap_df.sort_values("Impact", ascending=False) # order by impact

    st.bar_chart(shap_df.set_index("Feature"))




    # Explanation:
    
    st.subheader("Explanation")

    top_features = shap_df.head(3) 

    reasons = []
    for i, row in top_features.iterrows():
        feature = row["Feature"]
        impact = row["Impact"]

        if impact > 0:
            reasons.append(f"{feature} increases the risk of heart disease")
        else:
            reasons.append(f"{feature} decreases the risk of heart disease")

    explanation_text = "The prediction is mainly influenced by: " + ", ".join(reasons) + "."

    st.write(explanation_text)
    
    


if st.button("Show Model Performance"):
    
    y_test = joblib.load("results/y_test.pkl")
    y_pred = joblib.load("results/y_pred.pkl")

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy, precision, recall, f1]
    })

    st.subheader("Model Performance")
    st.table(metrics_df)



if st.button("Show Confusion Matrix"):

    cm = joblib.load("results/cm.pkl")

    st.subheader("Confusion Matrix")

    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["No Disease", "Disease"],
        y=["No Disease", "Disease"],
        colorscale="Blues"
    )

    st.plotly_chart(fig)







   
    

