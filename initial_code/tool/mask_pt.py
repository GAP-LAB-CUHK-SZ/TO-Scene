def random_crop(self, org_coord, org_feat, org_label):
        # min_x = np.min(org_coord[:,0])
        # max_x = np.max(org_coord[:,0])
        # min_y = np.min(org_coord[:,1])
        # max_y = np.max(org_coord[:,1])
        # print(org_coord)
        LEN_X = torch.max(org_coord[:,0]) - torch.min(org_coord[:,0])
        LEN_Y = torch.max(org_coord[:,1]) - torch.min(org_coord[:,1])
        
        CROP_X = LEN_X/30
        CROP_Y = LEN_Y/30

        t=torch.zeros(org_coord.shape[0])
        crop_ids = []
        for i in range(100):
            # crop_center = org_coord[np.random.randint(0,org_coord.shape[0]),:] 
            center_x = torch.min(org_coord[:,0]) + np.random.randint(0,30)* CROP_X
            center_y = torch.min(org_coord[:,1]) + np.random.randint(0,30)* CROP_Y
            # center_x = torch.min(org_coord[:,0]) + ((i*9)//30)* CROP_X
            # center_y = torch.min(org_coord[:,1]) + ((i*9)%30)* CROP_Y


            crop_ids.append(np.where((org_coord[:,0]<center_x+CROP_X) & (org_coord[:,0]>center_x-CROP_X)& (org_coord[:,1]<center_y+CROP_Y) & (org_coord[:,1]>center_y-CROP_Y) )[0])
        crop_ids = np.concatenate(crop_ids)
        # crop_ids = torch.cat(crop_ids,dim=0)
        t[crop_ids]+=1
        save_ids = torch.where(t==0)[0]
        partial_coord = org_coord[save_ids]
        partial_feat = org_feat[save_ids]
        partial_label = org_label[save_ids]

        crop_ids = torch.where(t!=0)[0]
        crop_coord = org_coord[crop_ids]
        crop_feat = org_feat[crop_ids]
        crop_label = org_label[crop_ids]

        # print('partial_coord',partial_coord.size())
        # print('org_coord',org_coord.size())
        # sio.savemat('sample_test.mat', {'partial_points':partial_coord.cpu().detach().numpy(), 'target_coor':org_coord.detach().cpu().numpy()})
        # exit(0)
        return partial_coord, partial_feat,partial_label,crop_coord, crop_feat, crop_label