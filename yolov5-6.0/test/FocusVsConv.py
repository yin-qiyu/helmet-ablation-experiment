import torch
from models.common import Focus, Conv
from utils.torch_utils import profile


focus = Focus(3, 64, k=3).eval()
conv = Conv(3, 64, k=6, s=2, p=2).eval()

# Express focus layer as conv layer
conv.bn = focus.conv.bn
conv.conv.weight.data[:, :, ::2, ::2] = focus.conv.conv.weight.data[:, :3]
conv.conv.weight.data[:, :, 1::2, ::2] = focus.conv.conv.weight.data[:, 3:6]
conv.conv.weight.data[:, :, ::2, 1::2] = focus.conv.conv.weight.data[:, 6:9]
conv.conv.weight.data[:, :, 1::2, 1::2] = focus.conv.conv.weight.data[:, 9:12]

# Compare
x = torch.randn(16, 3, 640, 640)
with torch.no_grad():
    # Results are not perfectly identical, errors up to about 1e-7 occur (probably numerical)
    assert torch.allclose(focus(x), conv(x), atol=1e-6)

# Profile
results = profile(input=torch.randn(16, 3, 640, 640), ops=[focus, conv, focus, conv], n=10, device=0)