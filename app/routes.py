from flask import request, jsonify, send_file, render_template,url_for
from app import app
from .generator import MelodyGenerator
from music21 import converter
import openai

# Initialize your MelodyGenerator
melody_generator = MelodyGenerator(model_path="app/model.h5")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/generate_melody', methods=['POST'])
def generate_melody():
    generator = MelodyGenerator()
    seed = generator.generate_seed_melody()
    print("seed:",seed)
    melody = generator.generate_melody(seed, 500, 64, 0.3)
    if len(melody) > 60:
        melody = melody[:60]
    filename = "mel.mid"
    generator.save_melody(melody,0.25,"midi",filename)
    note_count = 0
    for i in melody:
        if i != "_":
            note_count +=1
    print("melody:", melody)
    print("num_notes:",note_count)
    lyrics = generate_lyrics(note_count,melody)
    # Assuming save_melody saves the file in the `static` directory
    file_url = url_for('static', filename=f"melodies/{filename}")

    return jsonify({'melodyPath': file_url, 'lyrics': lyrics})

@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics(num_notes,melody):
    print(num_notes)
    openai.api_key = ''
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "user", "content": f"Take a look at the following midi notes: {melody}. At the end of the response, describe the melody style(atmosphere, mood, etc)in a few words inside a bracket. "
                                        f" Then, According to the style, Generate catchy lyrics for music hook that have {num_notes} syllables.(calculate the syllables word by word, generate again if different)"
                                        f"Only print lyrics with the melody style in a bracket."}
        ]
    )
    lyrics = response.choices[0].message['content']
    return lyrics
