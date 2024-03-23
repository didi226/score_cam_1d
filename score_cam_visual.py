from score_cam import ScoreCAM
def score_cam_visual(self,target_layer="avg_pool2",norm_saliency_map_=False,right_label=True):
        dataloader_test = self.test_set
        cam_all = list()
        self.model.eval()
        model = copy.deepcopy(self.model)
        predicted_list= list()
        exec("net_e= ScoreCAM(model.cpu(), model." + target_layer +")")
        net = locals()['net_e']
        for idx_batch in iter(dataloader_test):
            netout = self.model(idx_batch[0].to(torch.float32).to(self.device))
            predicted = torch.argmax(netout.cpu().data, 1).numpy()
            right_labels=list()
            #predicted_list.extend(predicted) 
            for i_batch_size in range(idx_batch[0].shape[0]):
                if right_label:
                       if idx_batch[1][i_batch_size] != predicted[i_batch_size]:
                           continue
                input_tensor = Variable((idx_batch[0][i_batch_size, :, :].unsqueeze(0)))
                # net=ScoreCAM(model.cpu(), model.avg_pool2)
                activation_map = net(input=input_tensor, class_idx=predicted[i_batch_size],
                                     norm_saliency_map_=norm_saliency_map_).numpy().squeeze()
                cam_all.append(activation_map)
                right_labels.append(predicted[i_batch_size])
            right_labels=np.array(right_labels)
            predicted_list.append(right_labels)
        cam_all=np.array(cam_all)
        predicted_list=np.array(right_labels)
        return cam_all,predicted_list