"""
Dependency: pyttsx3
Install: pip install pyttsx3
"""
import pyttsx3

text_to_read = "Hello, This is the SIE ASL team."
# Change to your own path.
text_file = "../text.txt"
filename = "hello_pyttsx3.mp3"
mp3_file = "speech_pyttsx3.mp3"


'''
Initialization
'''
engine = pyttsx3.init()


'''
Set properties
'''

#Set rate: default rate is 200.
rate = engine.getProperty('rate')
engine.setProperty('rate', rate - 50)

#Set voice: voices is a list of voice installed on the machine.
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)


'''
Read text
'''
engine.say(text_to_read)
engine.runAndWait()

'''
Read text from a file
'''
with open(text_file, 'r', encoding = 'utf8') as f:
    engine.say(f.read())

engine.runAndWait()

'''
Save sound as a mp3 file
'''
engine.save_to_file(text_to_read, filename)
with open(text_file, 'r', encoding = 'utf8') as f:
    engine.save_to_file(f.read(), mp3_file)

engine.runAndWait()