from __future__ import print_function
#import librosa
import soundfile as sf
import numpy as np
from numpy import float32,float64
import sys
import time

__version__ = '0.5.1'

before = time.time()
start = before

# TODO:
#   get wav file by wget
#   compress file - ffmpeg?
#   generate stream url (apache -> http cgi)

L = 0
R = 1

audio_path = 'Audio/sample.wav'

f0 = 146.0 # sys.argv[2]
f_diff = 2.0 # sys.argv[3]

#A = 0.5

def gen_cosine(frequency, length, sr=44100, phase=None):
    step = 1.0/sr
    if phase is None:
        phase = -np.pi * 0.5

    return np.cos((2 * np.pi * frequency * (np.arange(step * length, step=step))+phase),
                   dtype=np.float32)

def amp_mod(sig, amp_array, sr, fade_dur=1.0):
    ''' Amplitude Modulation based on the evelope array given 
        param:
            sig: original signal
            amp_array: an array of amplitude values (0.0~1.0) for each sample 
            * both should be numpy array with the same length
        return: 
               numpy array modulated
    '''
    # global sr
    # fade in and out
    fade_length = fade_dur*sr
    fade_length = int(fade_length)
    amp_array[0] = 0

    for idx in range(1,fade_length):
        factor_in = np.sin(0.5 * np.pi * idx / fade_length)
        factor_out = np.sin(0.5 * np.pi * idx / fade_length + np.pi) + 1
        amp_array[idx] = amp_array[idx] * factor_in
        amp_array[idx-fade_length] = amp_array[idx-fade_length] * factor_out

    return sig * amp_array


# Load a wav file
y, sr = sf.read(audio_path, dtype='float32')
y = y.T
#y, sr = librosa.load(audio_path,sr=None, mono=False)
gen_len =  len(y[L])
print ("Load",time.time()-before)
before = time.time()

# Generate sine waves with two different freq for the two channels
tone = np.ndarray(shape=(2,gen_len),dtype=float32)
tone[L] = gen_cosine(f0, gen_len)
tone[R] = gen_cosine(f0+f_diff, gen_len)
print ("generate sine",time.time()-before)
before = time.time()

# A temporary envelope array
envelope = np.ones(gen_len) * A

# AM
tone[L] = amp_mod(tone[L], envelope, sr, 1.0 )
tone[R] = amp_mod(tone[R], envelope, sr, 1.0 )
print ("AM",time.time()-before)
before = time.time()

# Mix signals together
y[L] = y[L] + tone[L]
y[R] = y[R] + tone[R]

#test_tone = np.asarray(y, dtype=float32)
#print time.time()-before

# librosa.output.write_wav('test.wav', y=y, sr=sr)
sf.write('out.wav', y.T, sr, 'PCM_16')
print "Time used: {}".format( time.time()-start )
