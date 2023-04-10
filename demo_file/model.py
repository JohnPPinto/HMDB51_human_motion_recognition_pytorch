import torch
import torchvision

def create_model(num_classes: int, seed: int = 42):
    """
    A function to create a model.
    Parameters:
        num_classes: int, A integer for toal number of classes.
        seed: int(default: 42), A random seed value.
    Returns: 
        model: A feature extracted model for video classification.
        transforms: A torchvision transform is returned which was used in the pretrained model.    
    """
    # Creating model, weights and transforms
    weights = torchvision.models.video.MViT_V2_S_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.video.mvit_v2_s(weights=weights)
    
    # Freezing the model layers
    for params in model.parameters():
        params.requires_grad = False
        
    # Changing the fully Conncected head layer
    torch.manual_seed(seed)
    dropout_layer = model.head[0]
    in_features = model.head[1].in_features
    model.head = torch.nn.Sequential(
        dropout_layer,
        torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    )
    return model, transforms
