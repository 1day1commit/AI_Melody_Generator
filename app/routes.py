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
    if len(melody) > 80:
        melody = melody[:80]
    filename = "mel_2.mid"  # This should be the actual generated file name
    generator.save_melody(melody,0.25,"midi",filename)  # Implement saving logic inside your MelodyGenerator
    # Load the MIDI file
    midi = converter.parse(filename)
    num_notes = len(midi.recurse().notes)
    lyrics = generate_lyrics(num_notes,melody)
    # Assuming save_melody saves the file in the `static` directory
    file_url = url_for('static', filename=f"melodies/{filename}")

    return jsonify({'melodyPath': file_url, 'lyrics': lyrics})

@app.route('/generate_lyrics', methods=['POST'])
def generate_lyrics(num_notes,melody):
    # prompt = f"Write a song lyrics for a melody with {num_notes} notes."
    openai.api_key = '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'
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo-preview",  # or another model name
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"For MIDI notes {melody},generate lyrics that have {num_notes} syllables. Only print the lyrics, without using quotation marks."}
        ]
    )
    lyrics = response.choices[0].message['content']
    return lyrics




#
# @app.route('/generate_melody', methods=['POST'])
# def generate_melody():
#     data = request.json
#     seed = data.get('seed', '')
#     num_steps = data.get('num_steps', 500)
#     max_sequence_length = data.get('max_sequence_length', 64)
#     temperature = data.get('temperature', 0.3)
#
#     melody = melody_generator.generate_melody(seed, num_steps, max_sequence_length, temperature)
#     melody_generator.save_melody(melody, file_name="generated_melody.mid")
#
#     # Depending on your setup, you might want to serve the file differently.
#     return send_file("generated_melody.mid", as_attachment=True)

#
# @app.route('/generate_lyrics', methods=['POST'])
# def generate_lyrics():
#     data = request.json
#     prompt = data.get('prompt', '')
#
#     # Ensure you've set your OpenAI API key in your environment variables or configure it here
#     openai.api_key = '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'
#
#     response = openai.Completion.create(
#         engine="text-davinci-003",  # or the latest GPT version
#         prompt=prompt,
#         temperature=0.5,
#         max_tokens=100,
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0
#     )
#
#     return jsonify({"lyrics": response.choices[0].text.strip()})
