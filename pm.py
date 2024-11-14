import os
import argparse
if __name__== '__main__':
    args=None
    parser=argparse.ArgumentParser()
    parser.add_argument("--balance",type=int,default=None,help='eval epoch')
    parser.add_argument("--epoch",type=int,default=None,help='eval epoch')
    parser.add_argument("--num_point",type=int,default=2048,help='num_point')
    parser.add_argument("--depth",type=int,default=3,help='depth')
    parser.add_argument("--emb_dim",type=int,default=16,help='emb_dim')
    parser.add_argument("--encoder_dim",type=int,default=64,help='encoder_dim')
    parser.add_argument("--in_channel",type=int,default=5,help='in channel')
    parser.add_argument("--group_size",type=int,default=16,help='emb_dim')
    parser.add_argument("--num_group",type=int,default=64,help='emb_dim')
    parser.add_argument("--gpu",type=str,default='2',help='gpu')
    parser.add_argument("--name",type=str,default='mamba_id',help='gpu')
    parser.add_argument("--eval", action='store_true', help="set noise")
    parser.add_argument("--cand",type=str,default='pi0_line_sako,gamma_line_sako,pi+_line_sako,e-_line_sako',help='cand')

    args=parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
import sys
import json
import yaml
import torch as th
assert th.cuda.is_available()
from collections import namedtuple
from torchinfo import summary
from torcheval.metrics import BinaryAUROC
from torcheval.metrics import MeanSquaredError
import subprocess
from typing import Optional
import numpy as np
import gc
import datetime
import h5py
from sklearn.utils import shuffle
import tqdm
import math
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
from functools import partial

#from timm.models.layers import trunc_normal_
#from timm.models.layers import DropPath
from timm.layers import trunc_normal_
from timm.layers import DropPath

from mamba_ssm import Mamba

optuna = {'is_optuna' : False, 'trial' : None}

models = {}

th.autograd.set_detect_anomaly(True)



def tanh01(x):
    x[:,:,:2].tanh_()
    return x


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data

class Block(th.nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=th.nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False, drop_path=0.
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        
        # drop path 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else th.nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (th.nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: th.Tensor, residual: Optional[th.Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (self.drop_path(hidden_states) + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(th.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

class Encoder(th.nn.Module):
    def __init__(self, input_channel, encoder_channel,num_feature):
        super().__init__()
        self.input_channel = input_channel
        self.encoder_channel = encoder_channel
        self.num_feature=num_feature
        self.first_conv = th.nn.Sequential(
            th.nn.Conv1d(input_channel, self.num_feature, 1),
            th.nn.BatchNorm1d(self.num_feature),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(self.num_feature,self.num_feature*2,1)
        )
        self.second_conv = th.nn.Sequential(
            th.nn.Conv1d(self.num_feature*4,self.num_feature*4,1),
            th.nn.BatchNorm1d(self.num_feature*4),
            th.nn.ReLU(inplace=True),
            th.nn.Conv1d(self.num_feature*4, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        bs, g, n, _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, self.input_channel)

        feature = self.first_conv(point_groups.transpose(2,1))
        feature_global = th.max(feature, dim=2, keepdim=True)[0]
        feature = th.cat([feature_global.expand(-1,-1,n), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = th.max(feature, dim=2, keepdim=False)[0]

        return feature_global.reshape(bs, g, self.encoder_channel)

class Group(th.nn.Module):
    def __init__(self, input_channel,num_group, group_size):
        super().__init__()
        self.input_channel = input_channel
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self,xyz):
        batch_size, num_points, _ = xyz.shape
        center = fps(xyz, self.num_group) # batch, num_group, input_channel
        #center = fps(xyz, num_points//self.num_group)
        _, idx = self.knn(xyz,center)# _ , batch, num_group, group_size
        idx_base = th.arange(0, batch_size, device=xyz.device).view(-1,1,1) * num_points
        #print(xyz.shape,center.shape,idx.shape,_.shape,idx_base.shape,num_points,self.num_group)
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size,self.input_channel).contiguous()
        #neighborhood = neighborhood.view(batch_size, num_points//self.num_group, self.group_size,self.input_channel).contiguous()
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, th.nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                th.nn.init.zeros_(module.bias)
    elif isinstance(module, th.nn.Embedding):
        th.nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                th.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with th.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

def create_block(
        d_model,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        layer_idx=None,
        drop_path=0.,
        device='cuda',
        dtype=None,
    ):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
      
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        th.nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
        drop_path=drop_path,
    )
    block.layer_idx = layer_idx
    return block

class MixerModel(th.nn.Module):
    def __init__(
            self,
            d_model: int,
            n_layer: int,
            ssm_cfg=None,
            norm_epsilon: float = 1e-5,
            rms_norm: bool = False,
            initializer_cfg=None,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_out_in_block: int = 0.,
            drop_path: int = 0.1,
            device='cuda',
            dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # self.embedding = th.nn.Embedding(vocab_size, d_model, **factory_kwargs)

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = th.nn.ModuleList(
            [
                create_block(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = (th.nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else th.nn.Identity()
        self.drop_out_in_block = th.nn.Dropout(drop_out_in_block) if drop_out_in_block > 0. else th.nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None):
    #def forward(self, input_ids, pos, inference_params=None):
        hidden_states = input_ids  # + pos
        residual = None
        #hidden_states = hidden_states + pos
        for layer in self.layers:
            #print("hidden_state",layer,hidden_states.shape)
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=inference_params
            )
            hidden_states = self.drop_out_in_block(hidden_states)
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states

class PointMODE(th.nn.Module):
    def __init__(self, config, **kwargs):
        super(PointMODE, self).__init__()
        self.config = config

        self.name = config.model.NAME
        self.trans_dim = config.model.trans_dim
        self.depth = config.model.depth
        self.cls_dim = config.model.cls_dim
        self.reg_dim = config.model.reg_dim

        self.group_size = config.model.group_size
        self.num_group = config.model.num_group
        self.encoder_dims = config.model.encoder_dims
        self.encoder_feature = config.model.encoder_feature
        self.input_channel = config.model.in_channel
        self.emb_dim = config.model.emb_dim
        if(self.emb_dim==0):
            self.emb_dim=None
        self.fine_dim = config.model.fine_dim
        self.last_dim = config.model.last_dim
        self.model_args=config.model._asdict()
        self.weight = th.nn.Parameter(th.ones(1+self.reg_dim)).to("cuda")

        self.num_point=config.num_point
        if(self.name=="Mamba"):
            self.group_divider = Group(input_channel=self.input_channel, num_group=self.num_group, group_size=self.group_size)
            self.encoder = Encoder(input_channel=self.input_channel, encoder_channel=self.encoder_dims, num_feature=self.encoder_feature)

        else:
            self.num_feature=self.encoder_feature
            self.encoder = th.nn.Sequential(
                th.nn.Linear(self.input_channel*self.num_point, self.num_feature),
                th.nn.BatchNorm1d(self.num_feature),
                th.nn.ReLU(inplace=True),
                th.nn.Linear(self.num_feature,self.encoder_dims),
                th.nn.BatchNorm1d(self.encoder_dims),
                th.nn.ReLU(inplace=True),
            )


        self.use_cls_token = False if not hasattr(self.model_args, "use_cls_token") else self.model_args['use_cls_token']
        self.drop_path = 0. if not hasattr(self.model_args, "drop_path") else self.model_args['drop_path']
        self.rms_norm = False if not hasattr(self.model_args, "rms_norm") else self.model_args['rms_norm']
        self.drop_out_in_block = 0. if not hasattr(self.model_args, "drop_out_in_block") else self.model_args['drop_out_in_block']

        if self.use_cls_token:
            self.cls_token = th.nn.Parameter(th.zeros(1, 1, self.trans_dim))
            self.cls_pos = th.nn.Parameter(th.randn(1, 1, self.trans_dim))
            trunc_normal_(self.cls_token, std=.02)
            trunc_normal_(self.cls_pos, std=.02)

        if(not self.emb_dim is None):
            self.pos_embed = th.nn.Sequential(
                th.nn.Linear(self.input_channel, self.emb_dim),
                th.nn.GELU(),
                th.nn.Linear(self.emb_dim, self.trans_dim)
            )
        if(self.name=="Mamba"):
            self.blocks = MixerModel(d_model=self.trans_dim,
                                 n_layer=self.depth,
                                 rms_norm=self.rms_norm,
                                 drop_out_in_block=self.drop_out_in_block,
                                 drop_path=self.drop_path)
        if(self.name=="MLP"):
            self.blocks = th.nn.Sequential(
                th.nn.Linear(self.trans_dim,self.trans_dim),
                th.nn.BatchNorm1d(self.trans_dim),
                th.nn.ReLU(inplace=True),
                th.nn.Dropout(0.3),
            )
        if(self.name=="Conv"):
            self.blocks = th.nn.Sequential(
                th.nn.Conv1d(self.trans_dim*self.input_channel,self.trans_dim,1),
                #th.nn.BatchNorm1d(self.trans_dim),
                th.nn.ReLU(inplace=True),
                th.nn.Conv1d(self.trans_dim,self.trans_dim,1),
                #th.nn.BatchNorm1d(self.trans_dim),
                th.nn.ReLU(inplace=True),
            )

        self.norm = th.nn.LayerNorm(self.trans_dim)

        self.HEAD_CHANEL = 1
        if self.use_cls_token:
            self.HEAD_CHANEL += 1
        self.head_finetune = th.nn.Sequential(
            th.nn.Linear(self.trans_dim * self.HEAD_CHANEL, self.fine_dim),
            th.nn.BatchNorm1d(self.fine_dim),
            th.nn.ReLU(inplace=True),
            th.nn.Dropout(0.3),
            #th.nn.Linear(self.fine_dim, self.fine_dim),
            #th.nn.BatchNorm1d(self.fine_dim),
            #th.nn.ReLU(inplace=True),
            #th.nn.Dropout(0.3),
        )
        self.head_tune = th.nn.Linear(self.fine_dim, self.last_dim)
        self.cls_head = th.nn.Linear(self.last_dim, self.cls_dim)
        #= th.nn.Linear(self.last_dim, self.reg_dim)
        self.reg_head0 = None
        self.reg_head1 = None
        self.reg_head2 = None
        self.reg_head3 = None
        self.reg_head4 = None
        self.reg_head5 = None
        if self.reg_dim>0:
          self.reg_head0=th.nn.Linear(self.last_dim, 1,device="cuda")
        if self.reg_dim>1:
          self.reg_head1=th.nn.Linear(self.last_dim, 1,device="cuda")
        if self.reg_dim>2:
          self.reg_head2=th.nn.Linear(self.last_dim, 1,device="cuda")
        if self.reg_dim>3:
          self.reg_head3=th.nn.Linear(self.last_dim, 1,device="cuda")
        if self.reg_dim>4:
          self.reg_head4=th.nn.Linear(self.last_dim, 1,device="cuda")
        if self.reg_dim>5:
          self.reg_head5=th.nn.Linear(self.last_dim, 1,device="cuda")

        self.build_loss_func()
        self.drop_out = th.nn.Dropout(config['drop_out']) if "drop_out" in config else th.nn.Dropout(0)

    def build_loss_func(self):
        self.loss_ce = th.nn.CrossEntropyLoss()
        self.loss_mse = th.nn.MSELoss()
        self.loss_sml1 = th.nn.SmoothL1Loss(beta=0.1)

    def get_losses(self, output, target,norm=False):
        losses= [self.loss_ce(output[0], target[0])]
        #loss2 = self.loss_sml1(output[1],target[1])
        if(len(target[1][0])>0):
            for i in range(len(target[1][0])):
                losses.append(self.loss_sml1(output[i+1].ravel(),target[1][:,i]))
        tm=[]
        if norm:
            for i in range(len(losses)):
                tm.append(th.max(th.abs(output[i])))
            tm_sum=sum(tm)
            for i in range(len(losses)):
                losses[i]=losses[i]*tm_sum/(tm[i]+1)
        return losses

    def gradnorm_loss(self,output,target,alpha=0.001):
        losses=self.get_losses(output,target)
        weighted_loss=[self.weight[i].detach()*losses[i] for i in range(len(self.weight))]
        total_loss=sum(weighted_loss)
        gw=[]
        for i in range(len(self.weight)):
            dl=th.autograd.grad(self.weight[i]*losses[i],self.head_tune.parameters(),retain_graph=True,create_graph=True)
            grad_norm = th.norm(th.stack([g.norm(2) for g in dl if g is not None]))
            #gw.append(th.norm(dl))
            gw.append(grad_norm)
        total_loss.backward(retain_graph=True)
        mean_grad_norm = sum(gw)/len(gw)
        for i in range(len(self.weight)):
            self.weight.data[i]*=(gw[i]/mean_grad_norm).pow(alpha)
            if(i==0):self.weight.data[i]=max(self.weight[i].item(),0.2)
            else:self.weight.data[i]=max(self.weight[i].item(),0.05)
        with th.no_grad():
            self.weight/=self.weight.sum()

        return total_loss

    def get_loss(self, output, target):
        losses=self.get_losses(output,target)
        total_loss=sum(losses)
        return total_loss

    def _init_weights(self, m):
        if isinstance(m, th.nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, th.nn.Linear) and m.bias is not None:
                th.nn.init.constant_(m.bias, 0)
        elif isinstance(m, th.nn.LayerNorm):
            th.nn.init.constant_(m.bias, 0)
            th.nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, th.nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                th.nn.init.constant_(m.bias, 0)

    def forward(self, pts):
        if(self.name=="Mamba"):
            neighborhood, center = self.group_divider(pts)
            group_input_tokens = self.encoder(neighborhood)  # B G N
            if(not self.emb_dim is None):
                pos = self.pos_embed(center)
            # reordering strategy
            center_bond=[]
            group_input_tokens_bond=[]
            pos_bond=[]
            for i in range(self.input_channel):
                center_bond.append(center[:, :, i].argsort(dim=-1)[:, :, None])
                group_input_tokens_bond.append(group_input_tokens.gather(dim=1, index=th.tile(center_bond[i], (1, 1, group_input_tokens.shape[-1]))))
                if(not self.emb_dim is None):
                    pos_bond.append(pos.gather(dim=1, index=th.tile(center_bond[i], (1, 1, pos.shape[-1]))))
            group_input_tokens = th.cat(group_input_tokens_bond, dim=1)
            x = group_input_tokens
            # transformer
            x = self.drop_out(x)
            if(not self.emb_dim is None):
                pos = th.cat(pos_bond, dim=1)
                x = self.blocks(x+pos)
            else:
                x = self.blocks(x)
            x = self.norm(x)
            concat_f = x[:, :].mean(1)
            ret = self.head_finetune(concat_f)
        else:
            pts=pts.view(pts.shape[0],-1)
            input_tokens = self.encoder(pts)
            x = self.blocks(input_tokens)
            ret = self.head_finetune(x)
        ret = self.head_tune(ret)
        ret1 = self.cls_head(ret)
        if(self.reg_head0 is not None):
          red0=None
          red1=None
          red2=None
          red3=None
          red4=None
          red5=None
          if(self.reg_head0 is not None):
              red0=self.reg_head0(ret)
          if(self.reg_head1 is not None):
              red1=self.reg_head1(ret)
          if(self.reg_head2 is not None):
              red2=self.reg_head2(ret)
          if(self.reg_head3 is not None):
              red3=self.reg_head3(ret)
          if(self.reg_head4 is not None):
              red4=self.reg_head4(ret)
          if(self.reg_head5 is not None):
              red5=self.reg_head5(ret)
          return ret1,red0,red1,red2,red3,red4,red5
        else:
          return ret1

from collections import namedtuple

def re_namedtuple(pd):
  if isinstance(pd,dict):
    for key in pd:
      if isinstance(pd[key],dict):
        pd[key] = re_namedtuple(pd[key])
    pod = namedtuple('cfg',pd.keys())
    bobo = pod(**pd)

  return bobo


net=None

if __name__== '__main__':
    name=args.name
    cand = args.cand.split(",")
    in_channel = args.in_channel
    #target = ['phi', 'theta', 'P']
    target = ['phi','theta','P','dep']
    config=f"""
NAME: {name}
cand: {cand}
batch_size: 64
target: {target}
num_point: {args.num_point}
model:
  NAME: PointMamba
  in_channel: {in_channel}
  cls_dim: {len(cand)}
  reg_dim: {len(target)}
  emb_dim: {args.emb_dim}
  depth: {args.depth}
  group_size: {args.group_size}
  num_group: {args.num_group}
  trans_dim: {args.encoder_dim}
  encoder_dims: {args.encoder_dim}
  encoder_feature: {args.encoder_dim}
  fine_dim: 256
  last_dim: 64
  rms_norm: False
  drop_path: 0.2
  drop_out: 0.1
"""
    if(args.eval):
        epoch=args.epoch
        if(epoch is None):
            tl=th.load(f'save/ckpt-{name}-best.pth')
            epoch='best'
        else:
            tl=th.load(f'save/ckpt-{name}-{epoch:03d}.pth')
        config=(tl['config'])
    cfg=re_namedtuple(yaml.safe_load(config))
    try:cand=cfg.cand
    except:pass
    batch_size=cfg.batch_size
    net=PointMamba(cfg)
    net=net.cuda()
    #summary(net,input_size=(32,2,1200),col_names=("input_size","output_size","num_params","params_percent","kernel_size","mult_adds","trainable",))
    summary(net,input_size=(batch_size,cfg.num_point,cfg.model.in_channel),depth=6,col_names=("input_size","output_size","num_params"))
    #balance=50000
    balance=args.balance
    if(not args.eval):
        print(name)
        optim = th.optim.AdamW(net.parameters(), lr=0.001)
        train_dataset=h5pyData(cand,is_train=True,balance=balance,config=cfg)
        val_dataset=h5pyData(cand,is_val=True,balance=balance,config=cfg)
        train_loader=th.utils.data.DataLoader(train_dataset,batch_size=batch_size,num_workers=4,shuffle=True)
        val_loader=th.utils.data.DataLoader(val_dataset,batch_size=batch_size,num_workers=4)
        metric=BinaryAUROC(device='cuda')
        metric2=MeanSquaredError(device='cuda')
        val_metric=BinaryAUROC(device='cuda')
        val_metric2=MeanSquaredError(device='cuda')
        #h5_name=f"data/{name}.h5py"
        #train_shape=train_generator.shape
        #hf = h5py.File(h5_name, 'w')
        #train_X=hf.create_dataset("train_X",shape=train_shape,maxshape=(100000,train_shape[1],train_shape[2]),dtype='float32',chunks=(64,train_shape[1],train_shape[2]))
        #train_Y=hf.create_dataset("train_Y",shape=(train_shape[0],2),maxshape=(100000,2),dtype='float32',chunks=(64,2))
        print(f"balance{train_dataset.balance} train {train_dataset.begin}-{train_dataset.end} validation {val_dataset.begin}-{val_dataset.end} ")
        print(f"train {len(train_dataset)} validation {len(val_dataset)}")
        best_loss=1.
        early_stop=0
        epoch_min=args.epoch_min
        for epoch in range(200):
            net.train()
            th.set_grad_enabled(True)
            n = 0
            tloss=0
            metric.reset()
            metric2.reset()
            bar = tqdm.tqdm(total=len(train_loader),nrows=2,leave=False)
            #for i in range(len(train_generator)):
            #for X,Y1 in train_loader:
            for X,Y1,Y2 in train_loader:
                X = X.to("cuda", non_blocking=True)
                Y1 = Y1.to("cuda", non_blocking=True)
                Y2 = Y2.to("cuda", non_blocking=True)
                optim.zero_grad()
                output = net(X)
                #print(output.shape)
                loss, acc = net.get_loss_acc(output,[Y1,Y2])
                _loss=loss
                metric.update(output[0][:].ravel(),Y1[:].ravel())
                metric2.update(output[1],Y2)
                _loss.backward()
                optim.step()
                n += len(output[0])
                tloss+=loss.item()*len(output[0])
                bar.set_description(f"epoch{epoch:4d} : loss=({tloss/n:.4f}), auc {metric.compute().item():.4f} mse {metric2.compute().item():.4f} entry:{n:6d} ")
                bar.update()
            gc.collect()
            train_metric=metric.compute().item()
            train_metric2=metric2.compute().item()

            # validataion
            th.set_grad_enabled(False)
            net.eval()
            val_metric.reset()
            val_metric2.reset()
            n_val=0
            vloss=0
            bar = tqdm.tqdm(total=len(val_loader),nrows=2)
            for X,Y1,Y2 in val_loader:
                X = X.to("cuda")
                Y1 = Y1.to("cuda")
                Y2 = Y2.to("cuda")
                output = net(X)
                loss, acc = net.get_loss_acc(output,[Y1,Y2])
                val_metric.update(output[0][:].ravel(),Y1[:].ravel())
                val_metric2.update(output[1],Y2)
                n_val+=len(output[0])
                vloss+=loss.item()*len(output[0])
                bar.set_description(f"epoch{epoch:4d} : loss({tloss/n:.4f}) val loss({vloss/n_val:.4f}), auc {train_metric:.4f} val auc {val_metric.compute().item():.4f} mse {train_metric2:.4f} val mse {val_metric2.compute().item():.4f}", True)
                bar.update()
            bar.set_description(f"epoch{epoch:4d} : loss({tloss/n:.4f}) val loss({vloss/n_val:.4f}), auc {train_metric:.4f} val auc {val_metric.compute().item():.4f} mse {train_metric2:.4f} val mse {val_metric2.compute().item():.4f}", True)
            early_stop+=1
            if(vloss/n_val<best_loss):
                early_stop=0
                best_loss=vloss/n_val
                th.save({'net':net.state_dict(),'optimizer':optim.state_dict(),'vloss':best_loss,'epoch':epoch,'metrics':metric.state_dict(),'config':config},f'save/ckpt-{name}-best.pth')
                continue
            if epoch % 10 == 9 and epoch<20:
                th.save({'net':net.state_dict(),'optimizer':optim.state_dict(),'vloss':vloss/n_val,'epoch':epoch,'metrics':metric.state_dict(),'config':config},f'save/ckpt-{name}-{epoch:03d}.pth')
            if(epoch>epoch_min and early_stop>4):break
            #bar.close()
    if(args.eval):
        net.load_state_dict(tl['net'])
        print("vloss",tl['vloss'])
        test_metric=BinaryAUROC(device='cuda')
        test_metric2=MeanSquaredError(device='cuda')
        th.set_grad_enabled(False)
        net.eval()
        test_dataset=h5pyData(cand,balance=balance,config=cfg)
        test_loader=th.utils.data.DataLoader(test_dataset,batch_size=batch_size,num_workers=4,shuffle=True)
        print(f"balance{test_dataset.balance} test {test_dataset.begin}-{test_dataset.end} {len(test_dataset)}")
        X,Y1,Y2,origin=test_dataset[0]
        print(X.shape,Y1.shape,Y2.shape)
        hf=h5py.File(f'save/{name}_{epoch}.h5py', 'w')
        #test_X=hf.create_dataset("test_X",shape=(len(test_dataset),X.shape[0],X.shape[1]),dtype='float32')
        test_Y1=hf.create_dataset("test_Y1",shape=(len(test_dataset),Y1.shape[0]),dtype='float32')
        test_Y2=hf.create_dataset("test_Y2",shape=(len(test_dataset),Y2.shape[0]),dtype='float32')
        test_O=hf.create_dataset("test_O",shape=(len(test_dataset),origin.shape[0]),dtype='int32')
        test_p1=hf.create_dataset("test_p1",shape=(len(test_dataset),Y1.shape[0]),dtype='float32')
        test_p2=hf.create_dataset("test_p2",shape=(len(test_dataset),Y2.shape[0]),dtype='float32')
        bar = tqdm.tqdm(total=len(test_loader),nrows=2)
        test_metric.reset()
        test_metric2.reset()
        tloss=0
        n_t=0
        for X,Y1,Y2,origin in test_loader:
            #test_X[n:n+len(X)]=X
            test_O[n_t:n_t+len(X)]=origin
            test_Y1[n_t:n_t+len(X)]=Y1
            test_Y2[n_t:n_t+len(X)]=Y2[:]
            X = X.to("cuda")
            Y1 = Y1.to("cuda")
            Y2 = Y2.to("cuda")
            #Y2 = Y2[:,2:3].to("cuda")
            output = net(X)
            loss, acc = net.get_loss_acc(output,[Y1,Y2])
            test_metric.update(output[0][:].ravel(),Y1[:].ravel())
            test_metric2.update(output[1],Y2)
            test_p1[n_t:n_t+len(X)]=th.nn.functional.softmax(output[0],dim=1).cpu().numpy()
            test_p2[n_t:n_t+len(X)]=output[1].cpu().numpy()
            tloss+=loss.item()*len(output[0])
            n_t+=len(output[0])
            bar.set_description(f"loss({tloss/n_t:.4f}) test auc {test_metric.compute().item():.4f} test mse {test_metric2.compute().item():.4f}", True)
            bar.update()
        print(f"loss=({tloss/n_t:.4f}), auc {test_metric.compute().item():.4f} mse {test_metric2.compute().item():.4f} entry:{n_t:6d} ")
        hf.close()
        print(f'save/{name}_{epoch}.h5py')
