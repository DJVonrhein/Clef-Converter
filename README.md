# Clef-Converter
Switching between reading bass clef and treble clef can be tricky. This program takes an image of sheet music and converts all bass clef music to treble, or vice versa. It even works for piano music which usually has both present.

This python project uses OpenCV to isolate the symbols present in sheet music in order to move the notes and accidentals and swap clefs for simple reading.

Instructions:

  From command line on your computer, do 
  
    git pull (this url)
    
  After the project files are downloaded, enter the project directory. You now need a python 3.7 environment with the correct dependencies installed. For simplicity, you can try it through pip:   
  
    pip install -r requirements.txt  

  Pytesseract is the one library in this project that has an extra step to add to your environment. It requires you to first install tesseract, either through github or homebrew.
    
  Once your environment is set correctly, run clefconverter.py from the command line with either command: 
  
    python3 clefconverter.py (input_image_name) (output_image_name) treble 
    
    or
    
    python3 clefconverter.py (input_image_name) (output_image_name) bass 

  
