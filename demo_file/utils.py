import torch
import torchvision

def preprocess_video(video: str):
    """
    A function to preprocess the video file before going into the model.
    Parameters: 
        video: str, A string for the video file path.
    Returns: selected_frame: torch.Tensor, A tensor of shape 'TCHW'.
    """
    # Reading the video file
    vframes, _, _ = torchvision.io.read_video(filename=video, pts_unit='sec', output_format='TCHW')
    vframes = vframes.type(torch.float32)
    vframes_count = len(vframes)
    
    # Selecting frames at certain interval
    skip_frames = max(int(vframes_count/16), 1)
    
    # Selecting the first frame
    selected_frame = vframes[0].unsqueeze(0)
    
    # Creating a new sequence of frames upto the defined sequence length.
    for i in range(1, 16):
        selected_frame = torch.concat((selected_frame, vframes[i * skip_frames].unsqueeze(0)))
    return selected_frame
