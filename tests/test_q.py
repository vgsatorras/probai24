import pytest 
import torch

from probai24.src.models.ddpm import DDPM
from probai24.src.models.gt.ddpm import DDPM as GT


@pytest.fixture
def ddpm() -> DDPM:
    return DDPM(None)


@pytest.fixture
def ddpm_gt() -> GT:
    return GT(None)


def test_mean(ddpm: DDPM, ddpm_gt: GT):
    
    z = torch.ones(2, 1)
    t = torch.ones(2, dtype=torch.long)
    q_mean = ddpm._q_mean(z, t)
    q_mean_gt = ddpm_gt._q_mean(z, t)
    assert q_mean.shape == q_mean_gt.shape, (
        f"Incorrect _q_mean shape. Expecting {q_mean_gt.shape}, got {q_mean.shape}"
    )
    assert torch.allclose(q_mean, q_mean_gt), (
        "_q_mean is incorrectly implemented"
    )
    

def test_std(ddpm: DDPM, ddpm_gt: GT):
    
    z = torch.ones(2, 1)
    t = torch.ones(2, dtype=torch.long)
    q_std = ddpm._q_std(z, t)
    q_std_gt = ddpm_gt._q_std(z, t)
    assert q_std.shape == q_std_gt.shape, (
        f"Incorrect _q_std shape. Expecting {q_std_gt.shape}, got {q_std.shape}"
    )
    assert torch.allclose(q_std, q_std_gt), (
        "_q_std is incorrectly implemented"
    ) 
