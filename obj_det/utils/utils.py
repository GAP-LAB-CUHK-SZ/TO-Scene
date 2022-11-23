import torch

def load_checkpoint_model(checkpoint_path):
    checkpoint=torch.load(checkpoint_path)
    new_dict=[]
    for key in checkpoint:
        if key == "net":
            new_dict = {}
            for weight_key in checkpoint[key].keys():
                if weight_key.startswith("module."):
                    k_ = weight_key[7:]
                    # k_=weight_key.replace("module.","")
                    new_dict[k_] = checkpoint[key][weight_key]
                else:
                    print(weight_key)
                    new_dict[weight_key] = checkpoint[key][weight_key]
    return new_dict