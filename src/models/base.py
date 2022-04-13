import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def save(self, path):
        torch.save(self.state_dict(), path)

        return

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint)

        return
