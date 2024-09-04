import os
import datetime
import whisper
import torch
from torch import cuda, Generator
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="whisper.transcribe")

# Main function
def transcribe(file_path, model=None, language=None, verbose=False):
    """
    Transcribes a single audio file using OpenAI's Whisper model.

    Args:
        file_path (str): Path to the audio file to transcribe.
        model (str, optional): Name of the Whisper model to use for transcription.
            Defaults to None, which uses the default model.
        language (str, optional): Language code for transcription. Defaults to None,
            which enables automatic language detection.
        verbose (bool, optional): If True, enables verbose mode with detailed information
            during the transcription process. Defaults to False.

    Returns:
        str: A message indicating the result of the transcription process.

    Raises:
        RuntimeError: If an invalid file is encountered, it will be skipped.

    Notes:
        - The function downloads the specified model if not available locally.
        - The transcribed text will be saved in a "transcriptions" folder
          within the directory of the provided audio file.

    """    
    # Check for GPU acceleration
    if cuda.is_available():
        Generator('cuda').manual_seed(42)
    else:
        Generator().manual_seed(42)
        
    # Load model
    model = whisper.load_model(model)
    
    # Extract title and path
    title = os.path.basename(file_path).split('.')[0]
    folder_path = os.path.dirname(file_path)
    
    if verbose:
        print(f'\nTrying to transcribe file named: {title}')
    
    try:
        result = model.transcribe(file_path, language=language, verbose=verbose)
        
        # Make folder if missing 
        transcription_folder = os.path.join(folder_path, 'transcriptions')
        os.makedirs(transcription_folder, exist_ok=True)
        
        # Create segments for text file
        start = []
        end = []
        text = []
        for segment in result['segments']:
            start.append(str(datetime.timedelta(seconds=segment['start'])))
            end.append(str(datetime.timedelta(seconds=segment['end'])))
            text.append(segment['text'])
            
        # Save file to transcriptions folder
        with open(os.path.join(transcription_folder, f"{title}.txt"), 'w', encoding='utf-8') as file:
            file.write(title)
            for i in range(len(result['segments'])):
                file.write(f'\n[{start[i]} --> {end[i]}]:{text[i]}')
                
        output_text = f'Transcription finished for {title}, output can be found in {transcription_folder}'
    
    # Handle invalid file errors
    except RuntimeError:
        output_text = 'Not a valid file, skipping.'
    
    # Return output text
    return output_text

# Sample usage
if __name__ == "__main__":
    file_path = 'sample_audio/I_have_a_dream.mp3'
    print(transcribe(file_path, model="base", language="en", verbose=True))


    