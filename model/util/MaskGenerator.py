import torch
import random
from enum import Enum

class FaultType(Enum):
    CHA = 1
    ROW = 2
    COL = 3

    def rand():
        r = random.randint(1, 3)
        if r == 1:
            return FaultType.CHA
        elif r==2:
            return FaultType.ROW
        else:
            return FaultType.COL

class MaskGenerator():
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def __generateFaultyMask(self, d1: int, d2: int, prob: float):
        """
            Returns a 2-D torch.Tensor object with shape (d1, d2) with cells set to
            1 with probability (1-prob) and to a random value in the range [-1, 1] with
            probability prob.

                Parameters:
                    d1: first dimension of the tensor
                    d2: second dimension of the tensor
                    prob: probability of a faulty cell
                
                Returns:
                    Faulty mask with 1 in positions without faults and a random value in [-1, 1] otherwise
        """
        mask = torch.rand(d1, d2).to(self.device)
        faulty = mask < prob
        notFaulty = mask >= prob
        mask[faulty] = 1
        mask[notFaulty] = 0
        temp = torch.rand(d1, d2).to(self.device) * 2.0 - 1
        mask *= temp
        mask[mask == 0.0] = 1
        return mask
        

    def generateMask(self, chas: int, rows: int, cols: int, prob: float, faultyIndexes: list, faultType: FaultType) -> torch.Tensor:
        """
            Returns a torch.Tensor object with cells set to 1 with probability (1-prob),
            and cells set to a random value in the range [-1, 1] with probabiliy prob, for
            those cells at index in the dimension specified by faultType contained in
            faultyIndexes. All the other cells are set to 1.
                
                Parameters:
                    chas: number of channels of the tensor,
                    rows: number of rows of the tensor,
                    cols: number of columns of the tensor,
                    prob: probability to get a faulty cell
                    faultyIndexes: list of indexes of the cha/row/col with the faults
                    faultType (FaultType): indicates wether the fault is on a row, a column or a channel

                Returns:
                    mask (torch.Tensor): a tensor with 1s in positions without faults and a random value in [-1, 1] otherwise
        """
        res = torch.ones(1, chas, rows, cols).to(self.device)
        if faultType == FaultType.CHA:
            for faultyIndex in faultyIndexes:
                mask = self.__generateFaultyMask(rows, cols, prob)
                res[0, faultyIndex, :, :] = mask
        elif faultType == FaultType.ROW:
            for faultyIndex in faultyIndexes:
                mask = self.__generateFaultyMask(chas, cols, prob)
                res[0, :, faultyIndex, :] = mask
        elif faultType == FaultType.COL:
            for faultyIndex in faultyIndexes:
                mask = self.__generateFaultyMask(chas, rows, prob)
                res[0, :, :, faultyIndex] = mask
        return res