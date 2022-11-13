import tensorflow as tf
import tensorflow_hub as hub

import math
import statistics

from IPython.display import Audio, Javascript
from scipy.io import wavfile

from pydub import AudioSegment

import music21

# THIS PROGRAM USES THE SPICE MODEL TO CONVERT HUM TO PITCH
model = hub.load("https://tfhub.dev/google/spice/2")

A4 = 440
C0 = A4 * pow(2, -4.75)

# Function that converts the user-created audio to the format that the model 
# expects: bitrate 16kHz and only one channel (mono).
EXPECTED_SAMPLE_RATE = 16000
def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
  audio = AudioSegment.from_file(user_file)
  audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
  audio.export(output_file, format="wav")
  return output_file

def output2hz(pitch_output):
  # Constants taken from https://tfhub.dev/google/spice/2
  PT_OFFSET = 25.58
  PT_SLOPE = 63.07
  FMIN = 10.0;
  BINS_PER_OCTAVE = 12.0;
  cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET;
  return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)

def hz2offset(freq):
  # This measures the quantization error for a single note.
  if freq == 0:  # Rests always have zero error.
    return None
  # Quantized note.
  h = round(12 * math.log2(freq / C0))
  return 12 * math.log2(freq / C0) - h

def quantize_predictions(group, ideal_offset):
  note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

  # Group values are either 0, or a pitch in Hz.
  non_zero_values = [v for v in group if v != 0]
  zero_values_count = len(group) - len(non_zero_values)

  # Create a rest if 80% is silent, otherwise create a note.
  if zero_values_count > 0.8 * len(group):
    # Interpret as a rest. Count each dropped note as an error, weighted a bit
    # worse than a badly sung note (which would 'cost' 0.5).
    return 0.51 * len(non_zero_values), "Rest"
  else:
    # Interpret as note, estimating as mean of non-rest predictions.
    h = round(
        statistics.mean([
            12 * math.log2(freq / C0) - ideal_offset for freq in non_zero_values
        ]))
    octave = h // 12
    n = h % 12
    note = note_names[n] + str(octave)
    # Quantization error is the total difference from the quantized note.
    error = sum([
        abs(12 * math.log2(freq / C0) - ideal_offset - h)
        for freq in non_zero_values
    ])
    return error, note


def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth,
                               prediction_start_offset, ideal_offset):
  # Apply the start offset - we can just add the offset as rests.
  pitch_outputs_and_rests = [0] * prediction_start_offset + \
                            pitch_outputs_and_rests
  # Collect the predictions for each note (or rest).
  groups = [
      pitch_outputs_and_rests[i:i + predictions_per_eighth]
      for i in range(0, len(pitch_outputs_and_rests), predictions_per_eighth)
  ]

  quantization_error = 0

  notes_and_rests = []
  for group in groups:
    error, note_or_rest = quantize_predictions(group, ideal_offset)
    quantization_error += error
    notes_and_rests.append(note_or_rest)

  return quantization_error, notes_and_rests

def main():
  # Converting to the expected format for the model
  # in all the input 4 input method before, the uploaded file name is at
  # the variable uploaded_file_name
  converted_audio_file = convert_audio_for_model('Mary2.wav')

  # Loading audio samples from the wav file:
  sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

  # NOW TO ACTUALLY RUN THE SPICE MODEL ON THE AUDIO FILE

  # We now feed the audio to the SPICE tf.hub model to obtain pitch and uncertainty outputs as tensors.
  model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))

  pitch_outputs = model_output["pitch"]
  uncertainty_outputs = model_output["uncertainty"]

  # 'Uncertainty' basically means the inverse of confidence.
  confidence_outputs = 1.0 - uncertainty_outputs

  confidence_outputs = list(confidence_outputs)

  pitch_outputs = [float(x) for x in pitch_outputs]

  indices = range(len (pitch_outputs))

  pitch_outputs_and_rests = [
      output2hz(p) if c >= 0.8 else 0
      for i, p, c in zip(indices, pitch_outputs, confidence_outputs)
  ]

  # The ideal offset is the mean quantization error for all the notes
  # (excluding rests):
  offsets = [hz2offset(p) for p in pitch_outputs_and_rests if p != 0]

  ideal_offset = statistics.mean(offsets)


  best_error = float("inf")
  best_notes_and_rests = None
  best_predictions_per_note = None

  for predictions_per_note in range(20, 65, 1):
    for prediction_start_offset in range(predictions_per_note):

      error, notes_and_rests = get_quantization_and_error(
          pitch_outputs_and_rests, predictions_per_note,
          prediction_start_offset, ideal_offset)

      if error < best_error:      
        best_error = error
        best_notes_and_rests = notes_and_rests
        best_predictions_per_note = predictions_per_note

  # At this point, best_notes_and_rests contains the best quantization.
  # Since we don't need to have rests at the beginning, let's remove these:
  while best_notes_and_rests[0] == 'Rest':
    best_notes_and_rests = best_notes_and_rests[1:]
  # Also remove silence at the end.
  while best_notes_and_rests[-1] == 'Rest':
    best_notes_and_rests = best_notes_and_rests[:-1]


  # Creating the sheet music score.
  sc = music21.stream.Score()
  # Adjust the speed to match the actual singing.
  bpm = 60 * 60 / best_predictions_per_note
  print ('bpm: ', bpm)
  a = music21.tempo.MetronomeMark(number=bpm)
  sc.insert(0,a)

  for snote in best_notes_and_rests:   
      d = 'half'
      if snote == 'Rest':      
        sc.append(music21.note.Rest(type=d))
      else:
        sc.append(music21.note.Note(snote, type=d))

  # Saving the recognized musical notes as a MIDI file
  converted_audio_file_as_midi = converted_audio_file[:-4] + '.mid'
  fp = sc.write('midi', fp=converted_audio_file_as_midi)