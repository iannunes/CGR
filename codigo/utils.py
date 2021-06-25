import numpy as np
import os
import shutil
import torch
from torch.nn import functional as F

def sample_gaussian(m, v):
	sample = torch.randn(m.shape).cuda()
	#sample = torch.randn(m.shape)
	#use_cuda = torch.cuda.is_available() and True
	#device = torch.device("cuda" if use_cuda else "cpu")
	#sample = torch.randn(m.shape).to(device)
	z = m + (v**0.5)*sample
	return z

def gaussian_parameters(h, dim=-1):
	m, h = torch.split(h, h.size(dim) // 2, dim=dim)
	v = F.softplus(h) + 1e-8
	return m, v

def kl_normal(qm, qv, pm, pv, yh):
	element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm - yh).pow(2) / pv - 1)
	kl = element_wise.sum(-1)
	#print("log var1", qv)
	return kl
