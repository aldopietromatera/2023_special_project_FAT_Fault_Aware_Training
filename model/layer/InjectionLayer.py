import torch
import torch.nn as nn
import random
from model.util.MaskGenerator import MaskGenerator, FaultType

device = torch.device('cuda')


class FaultInjectionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, prob, faults, injFwd=False, injBwd=False):
        # Save the context of prob and faults so that you can use it during backward
        ctx.save_for_backward(faults, torch.tensor([injBwd]))

        #       000     1
        #       001     1
        #       010     1
        #       011     1
        #       100     0
        #       101     1
        #       110     1
        #       111     1

        # Apply the mask
        if injFwd and prob != 0 and not (faults is None):
            return x * faults
        else:
            return x
        
        # wrong if clause
        # if not injFwd or prob == 0 or faults is None :
        #     return x
        # else:
        #     return x * faults

    @staticmethod
    def backward(ctx, grad_output):
        # Load the context saved during forward
        faults, injBwd = ctx.saved_tensors
        # multiply grad_output by faults
        if injBwd[0] and not (faults is None):
            grad_output *= faults
        return  grad_output,\
                torch.tensor([0], device=device),\
                torch.tensor([0], device=device),\
                None,\
                None


class InjectionLayer(nn.Module):
    __idx = 0
    __generator = MaskGenerator(device)

    def __init__(self, prob):

        super(InjectionLayer, self).__init__()

        self.prob = torch.nn.Parameter(torch.ones(1) * prob)
        self._fprob = prob
        self.manualMask = False
        self.injFwd = True
        self.injBwd = True
        self.init_mask()

        self.__idx = InjectionLayer.__idx
        InjectionLayer.__idx += 1

        #print("Instantiating Injection Layer #{} with prob = {}".format(self.__idx, prob))
        if prob < 0 or prob > 1:
            raise ValueError("Invalid prob {!r}, should be a float in [0, 1]"
                             .format(prob))

    def forward(self, x):
        if self.faults is None and not self.manualMask:
            # TODO: questa generazione non deve essere dentro il forward ma dentro __init__ (da fare quando il layer verra' aggiunto automatiamente e non manualmente)
            self.faults = self.__generator.generateMask(chas=x.shape[1],
                                                        rows=x.shape[2],
                                                        cols=x.shape[3],
                                                        prob=self._fprob,
                                                        faultyIndexes=[self._get_rand_tensor_index(x)],
                                                        faultType=self.faultType)
        return FaultInjectionFunction.apply(x, self.prob, self.faults, self.injFwd, self.injBwd)

    def init_mask(self):
        self.faults = None
        self.faultType = FaultType.rand()
        #print(self.faultType)

    def apply_mask(self, mask, faultType):
        self.faults = mask
        self.faultType = faultType
        # print("mask applied: ", mask)

    def clear_mask(self):
        self.faults = None
        self.faultType = None

    def set_manualMask(self, manualMask):
        self.manualMask = manualMask
    
    def _get_rand_tensor_index(self, x):
        return random.randint(0, x.shape[self.faultType._value_]-1)