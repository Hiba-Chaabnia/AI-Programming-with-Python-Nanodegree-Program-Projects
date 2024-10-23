import argparse
import json
import torch
from functions import load_checkpoint, process_image, get_device

def predict(image_path, model, class_to_idx, top_k=5, device='cpu'):
    model.to(device)
    model.eval()
    
    tensor_image = process_image(image_path)
    tensor_image = tensor_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(tensor_image)
        
    probabilities = torch.softmax(output, dim=1)
    top_probs, top_indices = probabilities.topk(top_k, dim=1)
 
    top_probs = top_probs.squeeze().cpu().numpy()  
    top_indices = top_indices.squeeze().cpu().numpy()  

    idx_to_class = {value: key for key, value in class_to_idx.items()}
    top_classes = [idx_to_class[i] for i in top_indices]
    
    return top_probs, top_classes  


def main():
    parser = argparse.ArgumentParser(description='Predicting the class of an image using a trained neural network.')
    parser.add_argument('image_path', type=str, default='flowers/test/27/image_06864.jpg',help='Path to the image')
    parser.add_argument('checkpoint', type=str,default='./checkpoint.pth', help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str,default='cat_to_name.json', help='Path to JSON file mapping categories to real names')
    parser.add_argument('--gpu', default=False, action='store_true', help='Use GPU for predicting')
    args = parser.parse_args()

    device = get_device(args.gpu)

    model, class_to_idx = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, class_to_idx, args.top_k, device)

    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    classes = [cat_to_name[cls] for cls in classes]

    for i in range(args.top_k):
        print(f"Class:{classes[i]}, Probability:{probs[i]:.3f}") 

if __name__ == '__main__':
    main()
