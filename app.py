import gradio as gr
import numpy as np
import pickle
from keras.models import Sequential

from tensorflow.keras.preprocessing.sequence import pad_sequences

# LOAD MODEL
model = Sequential()
model.load_weights("model.keras")

# LOAD TOKENIZER
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# SETTINGS
max_sequence_len = 15

# REVERSE WORD MAP
index_to_word = {
    index: word
    for word, index in tokenizer.word_index.items()
}

# PREDICTION FUNCTION
def predict_next_words(
    seed_text,
    next_words,
    temperature
):

    try:

        seed_text = seed_text.lower()

        for _ in range(next_words):

            token_list = tokenizer.texts_to_sequences(
                [seed_text]
            )[0]

            if len(token_list) == 0:
                return "Please enter valid text."

            token_list = pad_sequences(
                [token_list],
                maxlen=max_sequence_len - 1,
                padding='pre'
            )

            predicted_probs = model.predict(
                token_list,
                verbose=0
            )[0]

            predicted_probs = np.asarray(
                predicted_probs
            ).astype("float64")

            predicted_probs = (
                predicted_probs /
                np.sum(predicted_probs)
            )

            predicted_probs = np.power(
                predicted_probs,
                1 / temperature
            )

            predicted_probs = (
                predicted_probs /
                np.sum(predicted_probs)
            )

            top_k = 10

            top_indices = np.argsort(
                predicted_probs
            )[-top_k:]

            top_probs = predicted_probs[
                top_indices
            ]

            top_probs = (
                top_probs /
                np.sum(top_probs)
            )

            predicted_word_index = np.random.choice(
                top_indices,
                p=top_probs
            )

            output_word = index_to_word.get(
                predicted_word_index,
                ""
            )

            if output_word == "<OOV>":
                continue

            seed_text += " " + output_word

        return seed_text

    except Exception as e:

        return f"Error: {str(e)}"

# CUSTOM CSS
custom_css = """
body {
    background-color: #050816;
}

.gradio-container {
    max-width: 950px !important;
    margin: auto;
    padding-top: 20px;
    font-family: Consolas, monospace;
}

h1 {
    text-align: center;
    color: #00ffcc;
    font-size: 52px !important;
    text-shadow: 0 0 20px #00ffcc;
}

p {
    text-align: center;
    color: #9ca3af;
    font-size: 17px;
}

textarea {
    background-color: #0f172a !important;
    color: #00ffcc !important;
    border: 2px solid #00ffcc !important;
    border-radius: 10px !important;
}

button {
    background-color: #00ffcc !important;
    color: black !important;
    border-radius: 10px !important;
    border: none !important;
    height: 55px !important;
    font-size: 18px !important;
    font-weight: bold !important;
}
"""

# GRADIO UI
demo = gr.Interface(
    fn=predict_next_words,

    inputs=[
        gr.Textbox(
            label="⌨️ Input Text",
            placeholder="Enter text...",
            lines=3
        ),

        gr.Slider(
            5,
            30,
            value=15,
            step=1,
            label="📜 Words to Generate"
        ),

        gr.Slider(
            0.1,
            1.5,
            value=0.7,
            step=0.1,
            label="⚡ Creativity"
        )
    ],

    outputs=gr.Textbox(
        label="🤖 AI Output",
        lines=6
    ),

    title="💻 AI Next Word Predictor",

    description="""
    Deep Learning based Next Word Prediction System
    """,

    css=custom_css
)

demo.launch()