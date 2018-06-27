""" This module prepares midi file data and feeds it to the neural
    network for training. """

# Imports
import os
import glob
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import CuDNNLSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint      
        
    
def main():
    """ Runs module functions to get notes from the midi files, generate network inputs/outputs, create network, and train it.
    
    Args:
        None.
        
    Returns:
        None.
    """
    
    # Get a list of all the notes, rests, and chords in the midi files
    notes = get_notes()
    
    # Load the notes from all video games combined
    all_notes = pickle.load(open('data/all_notes', 'rb'))

    # Get number of unique notes, rests, and chords in the midi files
    n_vocab = len(set(all_notes))

    # Generate Network Inputs (list of lists containing note sequences)
    # Generate Network Outputs (list of single notes, which come after the note sequence of the Network Input)
    network_input, network_output = prepare_sequences(notes, all_notes, n_vocab)

    # Generate the Keras model with final dense layer having n_vocab number of nodes
    model = create_network(network_input, n_vocab)

    # Train the network 
    train_network(model, network_input, network_output)

    
def get_notes():
    """ Get all the notes and chords from the midi files in the midi songs folder, dumps notes in data folder.
    
    Args:
        None.
        
    Returns:
        notes (list): Flat list of all the notes from all the midi files.
    """
    
    # Create an empty list to store all the midi file notes
    notes = []

    # Loop through all the files in the midi song folder
    for file in glob.glob("songs/*.mid"):
        
        # Create music21.Score object from midi file
        midi = converter.parse(file)

        # Print the current midi file being parsed
        print("Parsing %s" % file)

        # Reset to None for each for-loop
        notes_to_parse = None

        # Create notes_to_parse for each midi file
        try:
            # Midi file has instrument parts
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except:
            # Midi file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        # Append each note in notes_to_parse to the notes list
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, note.Rest):
                notes.append('Rest')
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # Dump the notes list to a pickle file
    pickle.dump(notes, open('data/notes', 'wb'))

    return notes


def prepare_sequences(notes, all_notes, n_vocab):
    """ Prepare the sequences used by the Neural Network.
    
    Args:
        notes (list): Flat list of all the notes from the midi files from this game.
        all notes (set): sorted set of unique notes, rests, and chords in the midi files from all games.
        n_vocab (int): Number of unique notes, rests, and chords in the midi files from all games.
        
    Returns:
        network_input (np.ndarray): List of lists containing note sequences (each of length 'sequence_length').
        network_output (np.ndarray): List of single notes, which come after the note sequence of the Network Input.
    """
    
    # Length of note sequences to be created for model training
    sequence_length = 25
    
    # Get a sorted set of unique notes, rests, and chords in all_notes from all video games
    all_pitch_names = sorted(set(item for item in all_notes))

    # Create a dictionary to map note pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(all_pitch_names))

    # Create empty lists for note sequence inputs (many notes) and note sequence output (single note)
    network_input = []
    network_output = []

    # Create input sequences (of length 'sequence_length') and the corresponding outputs (of length one)
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Number of different input sequence patterns
    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    
    # Normalize the network input by dividing by n_vocab (number of unique notes, rests, and chords)
    network_input = network_input / float(n_vocab)

    # One-hot encodes the network output
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)


def create_network(network_input, n_vocab):
    """ Create the structure of the neural network.
    
    Args:
        network_input (list): List of lists containing note sequences (each of length 'sequence_length').
        n_vocab (int): Number of unique notes, rests, and chords in the midi files.
        
    Returns:
        model (keras.models.Sequential): Keras model.
    """
    
    # Create sequential Keras model
    model = Sequential()
    model.add(CuDNNLSTM(256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True))
    model.add(Dropout(0.3))
    model.add(CuDNNLSTM(256))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model


def train_network(model, network_input, network_output):
    """ Train the neural network, saving the model weights periodically.
    
    Args:
        model (keras.models.Sequential): Keras model.
        network_input (list): List of lists containing note sequences (each of length 'sequence_length').
        network_output (list): List of single notes, which come after the note sequence of the Network Input.
    
    Returns:
        None
    """
    
    # Create Keras callback to save the model weights whenever the model improves
    filepath = "weights/weights_{epoch:02d}_{loss:.4f}_bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min')
    callbacks_list = [checkpoint]

    # Delete all the current weights in the weights folder (before running)
    folder = 'weights'
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
    
    # Fit the model, saving the weights whenever it improves
    model.fit(network_input, network_output, epochs=300, batch_size=64, callbacks=callbacks_list)
    
    # Save the final weights
    model.save_weights("weights/weights_final.hdf5")

    
if __name__ == '__main__':
    main()
