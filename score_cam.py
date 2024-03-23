import torch
import torch.nn.functional as F



class ScoreCAM:
    """
        ScoreCAM, inherit from BaseCAM
    """
    def __init__(self,model,target_layer, use_cuda=False):
        self.model_arch = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        self.activations = dict()
        if self.cuda:
            self.model = model.cuda()

        def forward_hook(module, input, output):
                if torch.cuda.is_available():
                    self.activations['value'] = output.cuda()
                else:
                    self.activations['value'] = output
                return None
        self.target_layer.register_forward_hook(forward_hook)

    def forward(self, input, class_idx=None, retain_graph=False,norm_saliency_map_=False):
        b, c, w = input.size()

        # predication on raw input
        logit = self.model_arch(input).cuda()

        if class_idx is None:
            predicted_class = logit.max(1)[-1]
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            predicted_class = torch.LongTensor([class_idx])
            score = logit[:, class_idx].squeeze()

        logit = F.softmax(logit)

        if torch.cuda.is_available():
            predicted_class = predicted_class.cuda()
            score = score.cuda()
            logit = logit.cuda()

        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, u, v = activations.size()
        score_saliency_map = torch.zeros(1,  c)
        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        with torch.no_grad():
            for i in range(u):
                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(c), mode='linear', align_corners=False)#'nearest' bilinear

                if saliency_map.max() == saliency_map.min():
                    continue
                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                # how much increase if keeping the highlighted region
                # predication on masked input
                input_new=torch.zeros(b,c,w)
                for i_times in range(input.shape[2]):     #Note that if you focus on time, you need to swap the dimensions of the shape
                    input_new[:,:,i_times]= input[:,:,i_times]*norm_saliency_map.cpu()
                output = self.model_arch(input_new)
                output = F.softmax(output)
                score = output[0][predicted_class]
                score_saliency_map=score_saliency_map.cpu()
                score_saliency_map += score * torch.squeeze(saliency_map.cpu(), dim=1).cpu()
        if  norm_saliency_map_:
            score_saliency_map = 2 * ((score_saliency_map - score_saliency_map.min()) / (score_saliency_map.max() - score_saliency_map.min())) - 1
        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None
        score_saliency_map = (score_saliency_map - score_saliency_map_min).div(
            score_saliency_map_max - score_saliency_map_min).data

        return score_saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False,norm_saliency_map_=False):
        return self.forward(input, class_idx, retain_graph,norm_saliency_map_)