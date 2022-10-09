"""
Shaposhnik Vladislav 09.10.22
"""

import matplotlib.pyplot as plt
import numpy as np
import wave, math, contextlib
from pylab import ceil, arange, log10
from scipy.io import wavfile

# the .wav file to compresse
fname = 'wave.wav'

# the returned .wav file
outname = 'filtered.wav'

# color of the plot at the end 
color = 'r' # red

# la frequence de coupure (soit tout ce qui depasse va a la poubelle)
cutOffFrequency = 900.0

def fft_dis(fname):
    sampFreq, snd = wavfile.read(fname)

    snd = snd / (2. ** 15) #convert sound array to float pt. values

    s1, s2 = snd[0::2], snd[1::2] # left & right channels
    
    #get the lenght & take the fourier transform of left channel
    n, p = len(s1), np.fft.rfft(s1)
    
    #get the lenght & take the fourier transform of left channel
    m, p2 = len(s2), np.fft.rfft(s2)
    
    # coef de moyennage des frequences 
    taille = 15 # de base 2.0 ~ 5.0
    # get the 
    nUniquePts, mUniquePts = int(ceil((n + 1) / float(taille))), int(ceil((m + 1) / float(taille)))
    
    p, p2 = p[0:nUniquePts], p2[0:mUniquePts]
    p, p2 = abs(p), abs(p2)
    
    p, p2 = p / float(n), p2 / float(m) # scale by the number of points so that
    p = p ** 2
    # we've got odd number of points fft
    if n % 2 > 0:
        p[1:len(p)] = p[1:len(p)] * 2
    else:
        # we've got even number of points fft
        p[1:len(p) -1] = p[1:len(p) - 1] * 2
    """
    Si vous affectez un canal Surround frontal ou arrière gauche à un fichier source stéréo 
    Compressor redirige le fichier source sur le canal gauche (et ignore le canal droit).
    """          
    freqArray = arange(0, nUniquePts, 1.0) * (sampFreq / n);
    plt.plot(freqArray / cutOffFrequency, 10 * log10(p), color = color)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Power (dB)')
    plt.show()


def getFiltredArray(x, windowSize):
  cumsum = np.cumsum(np.insert(x, 0, 0)) 
  # on moyenne les frequences
  return (cumsum[windowSize:] - cumsum[:-windowSize]) / windowSize
    
def getTypeWave(raw_bytes, n_frames, n_channels, sample_width, interleaved = True):
    if sample_width == 1:
        dtype = np.uint8 # 8 bits 
    elif sample_width == 2:
        dtype = np.int16 # 16 bits => 2 bytes
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")
        
    channels = np.fromstring(raw_bytes, dtype=dtype)
    if interleaved:
        # channels are interleaved, Sample N of channel M follows sample N of channel M-1 in raw data
        channels.shape = (n_frames, n_channels)
        channels = channels.T
    else:
        # All samples from channel M occur before all samples from channel M-1
        channels.shape = (n_channels, n_frames)
    return channels

# 'rb' => for read only
# open the file fname and only read it
with contextlib.closing(wave.open(fname,'rb')) as originWave:
    # initialisation 
    waveRate, amplitudeWidth = originWave.getframerate(), originWave.getsampwidth()
    nChannels, nbFrames = originWave.getnchannels(), originWave.getnframes()
    
    # Extract Raw Audio from multi-channel Wav File
    signal = originWave.readframes(nbFrames * nChannels)
    channels = getTypeWave(signal, nbFrames, nChannels, amplitudeWidth)
    # get window size
    fqRatio = (cutOffFrequency / waveRate)
    N = int( math.sqrt( np.random.randint(0, 1) + fqRatio ** 2) / fqRatio )

    # Use moviung average (only on first channel)
    filt = getFiltredArray(channels[0], N).astype(channels.dtype)

    with wave.open(outname, "w") as filtredWave:
        filtredWave.setparams((1, amplitudeWidth, waveRate, nbFrames, originWave.getcomptype(), originWave.getcompname()))
        # the values are 'C' & 'F' for C language and Fortran language
        filtredWave.writeframes(filt.tobytes('C'))
        # the files are closed automaticaly at the end of the with

# montre le plot du fichier de base "wave.wav"
print("Fichier Avant ! {}".format(fname))
fft_dis(fname)
# montre le plot du fichier apres la compression "filtred.wav"
print("Fichier Apres ! {} ~{}".format(fname, outname))
fft_dis(outname)