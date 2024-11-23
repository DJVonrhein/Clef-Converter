# Clef-Converter
Switching between reading bass clef and treble clef can be tricky. This program takes an image of sheet music and converts all bass clef music to treble, or vice versa. It is even designed to handle piano music which usually has both present.

Piano sheet converted to treble clef example:

<img src="https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/obsessed.png" width=500>  <img src="https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/obsessed-to-treble.png" width=500>

Non-Piano sheeet converted to bass clef example:

<img src="https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/takefive.png" width=500> <img src="https://github.com/DJVonrhein/Clef-Converter/blob/main/examples/takefive-to-bass.png" width=500>

At times, words go undetected and end up being shifted by mistake. For readable results, however, the program expects MuseScore or similar quality sheets -- not old, photocopied, or off axis sheets.

Setup Instructions:

  First, download this repo. Either download it as a .ZIP or type this command:
  
    git clone https://github.com/DJVonrhein/Clef-Converter.git
    
  You now need a python 3.7+ environment with the correct dependencies installed. For simplicity, I recommend pip:   
  
    pip install -r requirements.txt  

  Pytesseract is the one library in this project that has an extra step to add to your environment. It requires you to first install tesseract, either through github or homebrew. Afterward, try the above command again if it failed before.
    
  Once your environment is set correctly, you can run the program, which is at src/clefconverter.py . From the src directory, these are the commands to run it:
  
    python3 clefconverter.py (input_image_patah) (output_image_path) treble 
    
    or
    
    python3 clefconverter.py (input_image_path) (output_image_path) bass 

  The last argument is your desired clef for the entire sheet. 


About the Project:

  The project mostly uses OpenCV. It isolates elements of the music in a specific order. This isolation is achieved by thresholding, erosion, template matching and a few other functions.

