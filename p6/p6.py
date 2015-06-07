#p6.py

import sys
import scipy.io.wavfile as wavfile
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal

def main():
    if(sys.argv[1] == '1'):
        x = [0, 1, 2, 3]
        y = [4, 5, 6]
        print multiply(x,y)
    if(sys.argv[1] == '2'):
        part2()
    if(sys.argv[1] == '3'):
        part3()


#More sounds
def part3():
    sampleRate, sentence = get_sound_data("sentence.wav")
    #print sampleRate
    
    data = sentence * 1.0 / np.max(np.abs(sentence))
    #print data
    
    conv_filter = np.zeros_like(data)
    conv_filter[0] = 1
    conv_filter[sampleRate*.25] = .5
    #print conv_filter
    #print conv_filter.shape

    conv_filter2 = np.zeros_like(data)
    conv_filter2[0] = .001
    conv_filter2[sampleRate*36/344] = 1/3.0
    conv_filter2[sampleRate*80/344] = 1/4.0
    conv_filter2[sampleRate*106/344] = 1/8.0


    echo1 = signal.fftconvolve(data, conv_filter)
    echo2 = signal.fftconvolve(data, conv_filter2)

    with open("echo1.wav", 'w') as f:
        wavfile.write(f, sampleRate, echo1)
    with open("echo2.wav", 'w') as f:
        wavfile.write(f, sampleRate, echo2)

#Soundssss
def part2():
    data = None
    if(sys.argv[2] == "ahh"):
        data = get_sound_data("ahh.wav")

    elif(sys.argv[2] == "eee"):
        data = get_sound_data("eee.wav")

    elif(sys.argv[2] == "mmm"):
        data = get_sound_data("mmm.wav")


    #Part A
    if(sys.argv[3] == 'a'):
        plt.ylabel('Signal', fontsize = 28)
        plt.xlabel('Time', fontsize = 28)    
        time = np.linspace(0, len(data[1])/data[0], num=len(data[1]))
        plt.plot(time, data[1])


    #Part B
    elif(sys.argv[3] == 'b'):
        plt.ylabel('Magnitude', fontsize = 28)
        plt.xlabel('Tuple Index', fontsize = 28)
        print data[1]

        fourier_mag = np.absolute(np.fft.fft(data[1]))
        print fourier_mag

        plt.plot(fourier_mag)

        #Find max frequency
        freqs = np.fft.fftfreq(len(fourier_mag))
        idx = np.argmax(fourier_mag)
        max_freq = freqs[idx]
        freq_in_hz = abs(max_freq * data[0])
        print freq_in_hz


    plt.title(sys.argv[2], fontsize = 36)
    plt.show()


def get_sound_data(filename):
    with open(filename, 'r') as f:
        sampleRate, data = wavfile.read(f)
        if len(data.shape) == 2 and data.shape[1] == 2:
            data = data[:,1]
    return (sampleRate, data)


#Uses FFT and inverse FFT to multiple 2 arrays of numbers
#each of those numbers represents a digit
def multiply(x, y):
    #Pad with 0s
    two_n = max(len(x), len(y)) * 2
    while(len(y) < two_n):
        y.append(0)
    while(len(x) < two_n):
        x.append(0)

    #Take 'convolution' using FFT and IFFT, then round and cast to ints
    conv = np.absolute(np.fft.ifft(np.fft.fft(x) * np.fft.fft(y)))
    conv = [int(round(x)) for x in conv]

    #convert back to format required by problem
    answer = 0
    for index, num in enumerate(conv):
        answer += np.power(10, index) * num
    return [int(i) for i in str(answer)][::-1]


if __name__ == "__main__":
    main()