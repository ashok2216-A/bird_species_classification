'''Copyright 2024 Ashok Kumar

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.'''



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
