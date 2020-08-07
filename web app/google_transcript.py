import speech_recognition as sr
from os import path
import glob
import sys

filetype = sys.argv[1]

AUDIO_FILE = path.join(path.dirname(path.realpath(__file__)), 'upload/uploaded_%s.wav' % filetype)

r = sr.Recognizer()

outputfile = open('res_google_%s.txt' % filetype, 'w', encoding='utf-8')

with sr.AudioFile(AUDIO_FILE) as source:
    audio = r.record(source)

try:
    # res = r.recognize_google(audio, language='vi-VN')
    res = r.recognize_sphinx(audio, language='vi-VN')
    # res = r.recognize_wit(audio, key="293915438362851")
    print('\n Google transcript: \n')
    print(res)
    outputfile.write(res)

except Exception as ex:
    print('\n Google transcript Exception: \n' + str(ex) + '\n')
    # pass
    outputfile.write(str(ex))
    pass

outputfile.close()