"""
This module is used to extract all of the notes from all of the different 
video game soundtracks. By using all the notes from all of the different 
video game soundtracks, the neural network weights can be transferred between 
video games to create all new sounds.
"""

# Imports
import glob
import pickle
from music21 import converter, instrument, note, chord


def main():
    """
    Runs module functions to extract all of the different notes used in 
    all of the video game soundtracks, in the form of a python list, which 
    is then pickled.
    
    Args:
        None.
    
    Returns:
        None.
    """
    
    # Collect all the song notes used in each video game
    get_pokemon_notes()
    get_zelda_notes()
    get_final_fantasy_notes()
    
    # Combine all the notes used in all three video games, pickle dump
    get_all_notes()


def get_pokemon_notes():
    """ 
    Get all the notes and chords from the midi music files from
    Pokemon Red/Blue/Yellow.
    
    Args:
        None.
        
    Returns:
        None.
    """
    
    # Create an empty list to store all the midi file notes
    notes = []

    # Loop through all the files in the midi song folder
    for file in glob.glob("pokemon_red_blue_yellow/songs/*.mid"):
        
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
    pickle.dump(notes, open('pokemon_red_blue_yellow/data/notes', 'wb'))


def get_zelda_notes():
    """ 
    Get all the notes and chords from the midi music files from
    Zelda Ocarina of Time.
    
    Args:
        None.
        
    Returns:
        notes (list): Flat list of all the notes from all the midi files.
    """
    
    # Create an empty list to store all the midi file notes
    notes = []

    # Loop through all the files in the midi song folder
    for file in glob.glob("zelda_ocarina_of_time/songs/*.mid"):
        
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
    pickle.dump(notes, open('zelda_ocarina_of_time/data/notes', 'wb'))


def get_final_fantasy_notes():
    """ 
    Get all the notes and chords from the midi music files from
    Final Fantasy 7.
    
    Args:
        None.
        
    Returns:
        notes (list): Flat list of all the notes from all the midi files.
    """
    
    # Create an empty list to store all the midi file notes
    notes = []

    # Loop through all the files in the midi song folder
    for file in glob.glob("final_fantasy_7/songs/*.mid"):
        
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
    pickle.dump(notes, open('final_fantasy_7/data/notes', 'wb'))


def get_all_notes():
    """
    Extracts all of the notes from all the different video game soundtracks.
    
    Args:
        None.
    
    Returns:
        None.
    """
    
    # Pickle load 'notes' list from each video game
    pokemon_notes = pickle.load(open( "pokemon_red_blue_yellow/data/notes", "rb"))
    zelda_notes = pickle.load(open("zelda_ocarina_of_time/data/notes", "rb"))
    final_fantasy_notes = pickle.load(open("final_fantasy_7/data/notes", "rb"))
    
    # Get sorted set of unique notes, rests, and chords in the midi files for each video game
    pokemon_set = sorted(set(item for item in pokemon_notes))
    zelda_set = sorted(set(item for item in zelda_notes))
    final_fantasy_set = sorted(set(item for item in final_fantasy_notes))
    
    # Combine video games, and create sorted set again
    all_notes = pokemon_set + zelda_set + final_fantasy_set
    all_notes = sorted(set(item for item in all_notes))
    
    # Pickle dump
    pickle.dump(all_notes, open("pokemon_red_blue_yellow/data/all_notes", "wb"))
    pickle.dump(all_notes, open("zelda_ocarina_of_time/data/all_notes", "wb"))
    pickle.dump(all_notes, open("final_fantasy_7/data/all_notes", "wb"))
    pickle.dump(all_notes, open("three_games_combined/data/all_notes", "wb"))
    

if __name__ == '__main__':
    main()