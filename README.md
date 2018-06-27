# video_game_music_generation
## Summary

Used recurrent neural networks to generate new MIDI music after training on soundtracks from my favorite childhood video games: Zelda Ocarina of Time, Pokemon Red/Blue/Yellow, and Final Fantasy VII.

Presentation summary can be found here: [Presentation](https://docs.google.com/presentation/d/1kI3YKBqnUK6ZKu0WqVaWVSTDeT_PAfFdWCS6GvjEqm0/edit?usp=sharing)

## Motivation

Music and video games are two of my favorite things in life. Together they can create truly magical experiences. I'll never forget the video game soundtracks accompanying choosing your first Pokemon from Professor Oak in Pokemon Red/Blue/Yellow, or finding the Kokiri Sword in the Lost Forest in Zelda Ocarina of Time. 

For this project I wanted to explore the music soundtracks from my favorite video games growing up: Zelda Ocarina of Time, Pokemon Red/Blue/Yellow, and Final Fantasy VII. I wanted to see if I could generate never before heard "video game music" using deep learning! It worked!

Other than my love for music and video games there is a business opportunity here. For AAA video games (highest development budgets) it can cost up to $2500 per minute of finished music. However, indie video game developers don't have this budget room and often have to make sacrifices with music quality. Hopefully projects like this can help indie video game developers create their dreams in the future!

## Repo Structure

- `get_all_notes.py` : Run this module first. This module extracts all of the notes from all of the different 
  video game soundtracks. This is needed so that all three video game's neural networks are trained on the same set of notes.
- `pokemon_red_blue_yellow/train.py` : This module trains a LSTM neural network on the MIDI song files from Pokemon Red/Blue/Yellow.
- `pokemon_red_blue_yellow/predict.py` : This module generates a new MIDI song file using the Pokemon Red/Blue/Yellow trained neural network, and a random song seed from Pokemon.
- `zelda_ocarina_of_time/train.py` : This module trains a LSTM neural network on the MIDI song files from Zelda Ocarina of Time.
- `zelda_ocarina_of_time/predict.py` : This module generates a new MIDI song file using the Zelda Ocarina of Time trained neural network, and a random song seed from Zelda.
- `final_fantasy_7/train.py` : This module trains a LSTM neural network on the MIDI song files from Final Fantasy 7.
- `final_fantasy_7/predict.py` : This module generates a new MIDI song file using the Final Fantasy 7 trained neural network, and a random song seed from Final Fantasy.
- `three_games_combined/train.py` : This module trains a LSTM neural network on the MIDI song files from Pokemon Red/Blue/Yellow, Zelda Ocarina of Time, and Final Fantasy 7.
- `three_games_combined/predict.py` : This module generates a new MIDI song file using the three-games-combined neural network, and a random song seed from any of the three games. 