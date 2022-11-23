import torch
from weighted_FPS_utils import weighted_furthest_point_sample

if __name__=="__main__":
    FPS_module=weighted_furthest_point_sample
    input=torch.randn((4,1000,4)).cuda()
    output=FPS_module(input,100,1,1)
    print(output.shape)