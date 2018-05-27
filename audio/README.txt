To convert your .wav file(s) into MFCCs that can be read by amdtk:


(1) Put the .wav file into this directory


(2) Edit the mfcc.conf file in this directory by changing the SOURCERATE value to match the .wav file. The equation to calculate the correct value of SOURCERATE is the following:

SOURCERATE = 10,000 / (khz of .wav file)

This is because "SOURCERATE" is actually the length of the period between samples, measured in 100nsec units. Yeah, it's annoying, but that's just how HCopy is programmed.

So, for example, for a 44.1 khz .wav file,

SOURCERATE = 10,000 / 44.1 = 226.8


(3) Run the following command from the audio directory:

HCopy -C mfcc.conf wav_file_name.wav desired_output_file_name.fea



There's also a way to bulk convert many .wav files by putting all the filenames in a text file, but I don't remember how to do that right now. You can probably learn by googling "HCopy bulk".

