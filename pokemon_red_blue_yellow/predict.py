""" This module generates notes for a midi file using the
    trained neural network. """

# Imports
import pickle
import numpy as np
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM


def main():
    """ Runs module functions to get notes from the midi files, generate network input (seed), and generate new midi files.
    
    Args:
        None.
        
    Returns:
        None.
    """
    
    # Load the notes used to train the model
    notes = pickle.load(open('data/notes', 'rb'))
    
    # Load the notes from all video games combined
    all_notes = pickle.load(open('data/all_notes', 'rb'))
    
    # Get number of unique notes, rests, and chords in the midi files
    n_vocab = len(set(all_notes))

    # Generate Network Inputs (list of lists containing note sequences)
    # Generate Normalized Network Input
    network_input, normalized_input = prepare_sequences(notes, all_notes, n_vocab)
    
    # Generate the Keras model with final dense layer having n_vocab number of nodes
    model = create_network(normalized_input, n_vocab)
    
    # Generate the note outputs from the model, and random sequence of notes for network input
    prediction_output = generate_notes(model, network_input, all_notes, n_vocab)
    
    # Create the Midi file from the generated note output
    create_midi(prediction_output)

    
def prepare_sequences(notes, pitch_names, n_vocab):
    """ Prepare the sequences used by the Neural Network.
    
    Args:
        notes (list): Flat list of all the notes from the midi files from this game.
        all notes (set): sorted set of unique notes, rests, and chords in the midi files from all games.
        n_vocab (int): Number of unique notes, rests, and chords in the midi files from all games.
    
    Returns:
        network_input (list): List of lists containing note sequences (each of length 'sequence_length').
        normalized_input (np.ndarray): Normalized numpy array containing note sequences.
    """
    
    # Length of note sequences to be created for model prediction seed
    sequence_length = 25
    
    # Create a dictionary to map note pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitch_names))

    # Create empty lists for note sequence inputs (many notes)
    network_input = []
    
    # Create input sequences (of length 'sequence_length')
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    # Number of different input sequence patterns
    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize the network input by dividing by n_vocab (number of unique notes, rests, and chords)
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)


def create_network(normalized_input, n_vocab):
    """ Create the structure of the neural network.
    
    Args:
        normalized_input (np.ndarray): Normalized numpy array containing note sequences.
        n_vocab (int): Number of unique notes, rests, and chords in the midi files.
    Returns:
        model (keras.models.Sequential): Keras model.
    """
    
    # Create sequential Keras model
    model = Sequential()
    model.add(CuDNNLSTM(256,
        input_shape=(normalized_input.shape[1], normalized_input.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(256))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    
    # Load the weights to each node
    model.load_weights('weights/weights_final.hdf5')

    return model


def generate_notes(model, network_input, pitch_names, n_vocab):
    """ Generate notes from the neural network based on a sequence of notes.
    
    Args:
        model (keras.models.Sequential): Keras model.
        network_input (list): List of lists containing note sequences (each of length 'sequence_length').
        pitch_names (dict): sorted set of unique notes, rests, and chords in the MIDI files.
        n_vocab (int): Number of unique notes, rests, and chords in the midi files.
    
    Returns:
        prediction_output (list): Generated sequence of notes, rests, and chords.
    """
    
    # Pick a random sequence from the input as a starting point for the prediction
    random_start = np.random.randint(0, len(network_input)-1)

    # Create a dictionary to map note integers to pitches
    int_to_note = dict((number, note) for number, note in enumerate(pitch_names))

    # Choose a random sequence of notes to use as the network seed
    pattern = network_input[random_start]
    
    # Create an empty list to record the prediction output for each successive note
    prediction_output = []

    # Generate 500 notes
    for num_notes in range(500):
        
        # Reshape the input into a format compatible with LSTM layers
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        
        # Normalize the network input by dividing by n_vocab (number of unique notes, rests, and chords)
        prediction_input = prediction_input / float(n_vocab)

        # Predict the next note, given an input sequence of notes
        prediction = model.predict(prediction_input, verbose=0)

        # Add each generated note to prediction_output
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # Update the pattern list to include to the generated note, used in next for-loop loop
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output


def create_midi(prediction_output):
    """ Convert the output from the prediction to notes and create a midi file from the notes.
    
    Args:
        prediction_output (list): Generated sequence of notes, rests, and chords.
        
    Returns:
        None.
    """
    
    # Set offset to zero
    offset = 0
    
    # Create an empty list to record music21 note, chord, and rest objects
    output_notes = []

    # Create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            # Pattern is a chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        elif pattern == 'Rest':
            # Pattern is a rest
            output_notes.append(note.Rest())
        else:
            # Pattern is a note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Increase offset each iteration so that notes do not stack
        offset += 0.5

    # Create music21 Stream object from the list of music21 notes, chords, and rests
    midi_stream = stream.Stream(output_notes)

    # Write the music21 Stream to a midi file
    midi_stream.write('midi', fp='generations/generated_song.mid')

    
if __name__ == '__main__':
    main()
