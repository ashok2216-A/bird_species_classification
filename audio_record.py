import pyaudio
import wave

class AudioRecorder:
    def __init__(self, output_filename, duration=5, sample_rate=44100, channels=2, chunk=1024, format=pyaudio.paInt16):
        self.output_filename = output_filename
        self.duration = duration
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk = chunk
        self.format = format
        self.audio = pyaudio.PyAudio()

    def record_audio(self):
        # Open recording stream
        stream = self.audio.open(format=self.format,
                                 channels=self.channels,
                                 rate=self.sample_rate,
                                 input=True,
                                 frames_per_buffer=self.chunk)

        print("Recording...")

        frames = []

        # Record for the specified duration
        for i in range(0, int(self.sample_rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Finished recording.")

        # Stop and close the stream
        stream.stop_stream()
        stream.close()

        # Save the recorded audio to a WAV file
        self.save_audio(frames)

    def save_audio(self, frames):
        with wave.open(self.output_filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))

    def terminate(self):
        # Terminate the audio instance
        self.audio.terminate()

# Example usage
# output_filename = "recorded_audio.wav"
# recorder = AudioRecorder(output_filename, duration=5)
# recorder.record_audio()
# recorder.terminate()

# print(f"Audio recorded and saved as {output_filename}")
