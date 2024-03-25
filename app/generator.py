import json
import os
import numpy as np
import tensorflow.keras as keras
import music21 as m21
import random
from .preprocess import SEQUENCE_LENGTH, MAPPING_PATH
from flask import current_app

class MelodyGenerator:
    """A class that wraps the LSTM model and offers utilities to generate melodies."""

    def __init__(self, model_path="app/model.h5"):
        """Constructor that initialises TensorFlow model"""

        self.model_path = model_path
        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH


    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        """Generates a melody using the DL model and returns a midi file.
        :return melody (list of str): List with symbols representing a melody
        """
        # create seed with start symbols
        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        # map seed to int
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            # limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            # one-hot encode the seed
            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))
            # (1, max_sequence_length, num of symbols in the vocabulary)
            onehot_seed = onehot_seed[np.newaxis, ...]

            probabilities = self.model.predict(onehot_seed)[0]
            output_int = self._sample_with_temperature(probabilities, temperature)
            recent_notes = []
            for _ in range(num_steps):
                ...
                output_int = self._sample_with_temperature(probabilities, temperature)

                # Check for repetition and adjust temperature dynamically
                if output_int in recent_notes[-5:]:
                    temperature = min(temperature + 0.1, 1.0)  # Increase diversity
                else:
                    temperature = max(temperature - 0.1, 0.3)  # Return to normal
                recent_notes.append(output_int)
                if len(recent_notes) > 10:  # Keep only the last 10 notes for checking
                    recent_notes.pop(0)

            # update seed
            seed.append(output_int)

            # map int to our encoding
            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]
            consecutive_underscores = 0
            if output_symbol == "_":
                consecutive_underscores += 1
                if consecutive_underscores > 4:
                    continue  # Skip appending the underscore and proceed to the next iteration
            else:
                consecutive_underscores = 0
            # check whether we're at the end of a melody
            if output_symbol == "/":
                break

            # update melody
            melody.append(output_symbol)

        return melody

    def generate_seed_melody(self):
        """ generate random seed melodies which will be the base of the generated melody (normally 3-5 notes)"""
        self.keys = [
        "53", "66", "87", "67", "68", "69", "84", "89", "60", "70", "83",
        "64", "61", "94", "81", "73", "63", "47", "48", "90", "93", "86", "59",
        "88", "75", "58", "45", "56", "79", "62", "71", "65", "78", "80", "50",
        "76", "54", "96", "85", "51", "91", "57", "72", "52", "46", "77", "49",
        "82", "55", "74"
        ]
        seed_elements = [random.choice(self.keys) + " "]  # Start with a note and a space
        total_length = random.randint(25, 30)

        while True:
            num_underscores = " ".join("_" * random.randint(1, 3)) + " "
            seed_elements.append(num_underscores)
            current_length = len(''.join(seed_elements))
            potential_next_note = random.choice(self.keys) + " "
            # Check if adding another note exceeds total_length
            if current_length + len(potential_next_note) <= total_length:
                seed_elements.append(potential_next_note)
            else:
                break

        seed_string = ''.join(seed_elements).strip()
        if len(seed_string) > total_length:
            seed_string = seed_string[:total_length].rstrip()

        return seed_string
    def _sample_with_temperature(self, probabilites, temperature):
        """Samples an index from a probability array reapplying softmax using temperature

        """
        predictions = np.log(probabilites) / temperature
        probabilites = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilites)) # [0, 1, 2, 3]
        index = np.random.choice(choices, p=probabilites)

        return index


    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):
        """Converts a melody into a MIDI file
        """
        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        # parse all the symbols in the melody and create note/rest objects
        for i, symbol in enumerate(melody):
            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):
                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1
                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        save_path = os.path.join(current_app.root_path, 'static', 'melodies',file_name)

        # write the m21 stream to a midi file
        stream.write('midi', fp=save_path)


if __name__ == "__main__":
    print("current_path:",current_app.root_path)
    mg = MelodyGenerator()
    seed = mg.generate_seed_melody()
    print("seed1", seed)

    melody = mg.generate_melody(seed,500 , SEQUENCE_LENGTH, 0.3)
    # seed = "67 _ 64 _ 67 _ _ 65_65 _ 64 _ 64 _ _",
    if len(melody) > 100:
        melody = melody[:100]

    print(len(melody))
    print(melody)
    mg.save_melody(melody)