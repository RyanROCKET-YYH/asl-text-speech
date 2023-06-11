"""
Dependency: gTTS
Install: pip install gTTS
"""

from gtts import gTTS
import os

'''Options'''
filename = "hello_gtts.mp3"
text_file="C:/Users/ah/Desktop/text.txt" # Change to your own path
mp3_file = 'speech_gtts.mp3'
text_to_read = "Hello, this is the SIE ASL team."
slow_audio_speed = False
language_en = 'en'

"""
Reading from a string
"""
def reading_from_string(str, language):
    tts = gTTS(text=str, lang=language, slow=slow_audio_speed)
    tts.save(filename)
    os.system(f'start {filename}')

"""
Reading from a text file
"""
def reading_from_file(input_file, language, output_file):
    # Read the text from the input file
    with open(text_file, 'r') as file:
        text = file.read().replace('\n', '')

    # Create a gTTS object and specify the language
    tts = gTTS(text=text, lang=language_en)

    # Save the speech as an MP3 file
    tts.save(output_file)
    os.system(f'start {output_file}')


if __name__ == '__main__':
    # Convert text to speech
    reading_from_string(text_to_read, language_en)
    reading_from_file(text_file, language_en, mp3_file)
