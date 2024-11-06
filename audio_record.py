import pyaudio
import wave

def record_audio(output_filename, duration=5, sample_rate=44100, channels=2, chunk=1024, format=pyaudio.paInt16):
    audio = pyaudio.PyAudio()

    # Open recording stream
    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording...")

    frames = []

    # Record for the specified duration
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio to a WAV file
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# output_filename = "recorded_audio.wav"
# record_audio(output_filename, duration=5)
# print(f"Audio recorded and saved as {output_filename}")

# Example usage
# output_filename = "recorded_audio.wav"
# record_audio(output_filename, duration=5)
# print(f"Audio recorded and saved as {output_filename}")
