
import gradio as gr
import numpy as np
import joblib

# Trained Titanic model load karo
model = joblib.load("titanic_model.pkl")

def predict_survival(pclass, sex, age, sibsp, parch, fare, embarked):
    # user input ko numpy array me convert karo
    features = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])
    pred = model.predict(features)
    return "✅ Survived" if pred[0] == 1 else "❌ Not Survived"

demo = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Passenger Class (1/2/3)"),
        gr.Radio([0,1], label="Sex (0 = Male, 1 = Female)"),
        gr.Number(label="Age"),
        gr.Number(label="Siblings/Spouse Aboard"),
        gr.Number(label="Parents/Children Aboard"),
        gr.Number(label="Fare"),
        gr.Radio([0,1,2], label="Embarked (0=S, 1=C, 2=Q)")
    ],
    outputs="text",
    theme="gradio/monochrome"
)

if __name__ == "__main__":
    demo.launch()
