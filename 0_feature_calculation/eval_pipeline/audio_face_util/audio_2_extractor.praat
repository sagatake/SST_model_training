# https://id.fnshr.info/2013/01/26/praatmaster/
# http://speechresearch.fiw-web.net/41.html#xf9054c6

# echo Hello world
# printline Goodbye world

# a=7.8
# b$="hoge"
# c$=" hoge?"
# d$ = b$ + c$

# printline 'd$'

# Praat Intro

#input_directory$ = "/Users/takeshi-s/Documents/research/experiment/4_pipeline/wav"
#output_directory$ = "/Users/takeshi-s/Documents/research/experiment/4_pipeline/out" 

#MARK_INPUT
input_directory$ = "../../src_audio"

#MARK_OUTPUT
output_directory$ = "temp_audio_out"

Create Strings as file list... list 'input_directory$'/*.wav
number_files = Get number of strings

for i from 1 to number_files
	selectObject("Strings list")
	current_token$ = Get string... 'i'
	Read from file... 'input_directory$'/'current_token$'
	info$ = Info
	objectname$ = extractLine$(info$, "Object name:")

	# Energy
	nocheck selectObject("Sound'objectname$'")
	nocheck energy$ = nocheck Get energy... 0.0 0.0
	nocheck energy$ > 'output_directory$'/'objectname$'_energy.csv

	# Pitch
	nocheck selectObject("Sound'objectname$'")
	nocheck To Pitch... 0.0 100 600.0
	Save as text file... 'output_directory$'/'objectname$'_pitch.csv
	
	# Intensity
	nocheck selectObject("Sound'objectname$'")
	nocheck To Intensity... 100.0 0.0
	nocheck Save as text file... 'output_directory$'/'objectname$'_intensity.csv

	# Formant
	nocheck selectObject("Sound'objectname$'")
	nocheck To Formant (burg)... 0.0 3.0 5500.0 0.0025 50.0
	nocheck Save as text file... 'output_directory$'/'objectname$'_formant.csv

	# Jitter and Shimmer
	nocheck selectObject("Sound'objectname$'")
	nocheck To PointProcess (periodic, cc)... 100.0 600.0
	nocheck selectObject("Sound'objectname$'","Pitch'objectname$'", "PointProcess'objectname$'")
	nocheck voiceReport$ = nocheck Voice report... 0 0 75 500 1.3 1.6 0.03 0.45
	nocheck voiceReport$ > 'output_directory$'/'objectname$'_voice_report.csv

	# Object removal to release allocated memory
	nocheck selectObject("Sound'objectname$'")
	nocheck Remove
	nocheck selectObject("Pitch'objectname$'")
	nocheck Remove
	nocheck selectObject("Intensity'objectname$'")
	nocheck Remove
	nocheck selectObject("Formant'objectname$'")
	nocheck Remove
	nocheck selectObject("PointProcess'objectname$'")
	nocheck Remove


endfor
# Play
# View & Edit

selectObject("Strings list")
Remove