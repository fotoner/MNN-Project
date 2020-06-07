from pydub import AudioSegment
import os

def wav_converter(src, dst):
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


def beat_to_bpm(beat):
    return 1.0 / beat * 60 * 1000


if __name__=='__main__':
    wav_converter(os.path.join(os.path.dirname('C:/osu!/Songs/144959 SHK - Identity Part 4/'), 'Identity Part 4.mp3'), './temp.wav')