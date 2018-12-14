import sunau
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from time import time
from python_speech_features import mfcc
from librosa import feature

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def read_data(genre):
    for i in range(100):
        name = 'genres/{}/{}.{:05d}.au'.format(genre, genre, i)

        with open("data/{}{}.csv".format(genre, i), "w") as new_file:
            f = sunau.Au_read(name)
            audio_data = list(np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16))

            for num in audio_data:
                new_file.write(str(num) + '\n')



def read_data_mfcc(genre):
    for i in range(100):
        name = "genres/{}/{}.{:05d}.au".format(genre, genre, i)

        with open("mfcc_data/{}{}.csv".format(genre, i), 'w') as data_file:
            f = sunau.Au_read(name)
            audio_data = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
            melfcc = mfcc(audio_data, samplerate=22050, numcep=6).flatten()
            for num in melfcc:
                data_file.write(str(num) + '\n')


def read_data_chroma(genre):
    for i in range(100):
        with open("data/{}{}.csv".format(genre, i), 'r') as input_file:
            data = np.array([float(x.strip()) for x in input_file.readlines()])

            with open("chroma_data/{}{}.csv".format(genre, i), 'w') as data_file:
                chroma = feature.chroma_stft(y=data, n_fft=256, n_chroma=12).flatten()
                for num in chroma:
                    data_file.write(str(num) + '\n')


if __name__ == '__main__':
    begin = time()
    with Pool(len(genres)) as p:
        p.map(read_data_chroma, genres)

    print("parallel Time: " + str(time() - begin))
