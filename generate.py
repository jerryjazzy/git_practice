from __future__ import print_function
import librosa
import numpy as np
from numpy import float32,float64
import sys
import time
import subprocess as sp

before = time.time()
start = before

L = 0
R = 1

audio_path = sys.argv[1]

f0 = 146.0 # sys.argv[2]
f_diff = 2.0 # sys.argv[3]

A = 0.5

def gen_cosine(frequency, length, sr=44100, phase=None):
    step = 1.0/sr
    if phase is None:
        phase = -np.pi * 0.5

    return np.cos((2 * np.pi * frequency * (np.arange(step * length, step=step))+phase),
                   dtype=np.float32)

def amp_mod(sig, amp_array, fade_dur=1.0):
    ''' Amplitude Modulation based on the evelope array given 
        param: sig should be a numpy array
               amp_array is a list with the same length as sig
        return: 
               numpy array modulated
    '''
    global sr
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

# Load the audio
y, sr = librosa.load(audio_path,sr=None, mono=False, duration=None)
gen_len =  len(y[R]) - 1
print ("Load",time.time()-before)
before = time.time()

# Generate the sine signal with differenty frequency for the two channels
tone = np.ndarray(shape=(2,gen_len),dtype=float32)
tone[L] = gen_cosine(f0, gen_len)
tone[R] = gen_cosine(f0+f_diff, gen_len)
print ("Generate Sine",time.time()-before)
before = time.time()

# Get envelope by calculating RMS of short-term energy
hop_length = sr/2 #sr/2
frame_length = hop_length * 2

env = librosa.feature.rmse(y=y, frame_length=frame_length, hop_length=hop_length)
print ("Calculate Envelope",time.time()-before)
before = time.time()

env_array = np.zeros(gen_len, dtype=float32)
for idx,energy in enumerate(env[0]):
    offset = idx * hop_length
    energy = energy * 0.85 + 0.15
    #energy = energy *0.70 + 0.30
    try:
        #env_array[offset:offset+hop_length] = [energy] * hop_length 
        env_array[offset:offset+hop_length] = np.full(hop_length, energy, dtype=float32)
    except:
    #    print 'last offset: %d' % offset
        last_len = len(env_array[offset:])
    #    print 'last length: %d' % last_len
        env_array[offset:] = np.full(last_len, energy, dtype=float32)

print ("Calculate Envelope Array",time.time()-before)
before = time.time()

# Amplitude Modulate tones
tone[L] = amp_mod(tone[L], env_array, fade_dur=1.0 )
tone[R] = amp_mod(tone[R], env_array, fade_dur=1.0 )
print ("AM",time.time()-before)
before = time.time()

# Mix signals together
y[L][:-1] = y[L][:-1] + tone[L]
y[R][:-1] = y[R][:-1] + tone[R]

# Apply a limiter to tackle clipping
#limiter = Limiter(attack_coeff, release_coeff, delay, threshold)
#y[L] = limitter.limit(y[L])
#y[R] = limiiter.limit(y[R])

# Write audio file
librosa.output.write_wav('test.wav', y , sr=sr, norm=True)
print ("done.")
#cmd = "./ffmpeg -v quiet -i test.wav -acodec pcm_s16le -vn -y out.wav"
#ret = None
#ret = sp.call(cmd, shell=True)
#print ("Ret: {}, Time used: {}".format( ret, time.time()-start ))

# Finally encode wav to m4a (could be commented out)
#sp.call("./ffmpeg -i out.wav -c:a aac -q:a 2 -y binaural_out.m4a", shell=True)

