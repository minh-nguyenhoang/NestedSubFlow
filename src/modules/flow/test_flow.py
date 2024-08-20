from typing import Optional
import warnings
from matplotlib.pylab import cond
import torch 
import torch.nn as nn
import torch.nn.functional as F
from .memcnn import InvertibleModuleWrapper
from ..tlc import LocalHalfInstanceNorm2d


class ChannelMasking:
    @staticmethod
    def split(x: torch.Tensor):
        x, y = torch.chunk(x,2, dim = 1)
        return x, y
    @staticmethod
    def merge(x: torch.Tensor, y: torch.Tensor):
        return torch.cat([x, y], dim=1)

class SpatialMasking:
    @staticmethod
    def split(x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(b, c, h//2, 2, w//2, 2).permute(0,1,2,4,3,5).flatten(-2)

        x, y = x[..., [0,3]], x[..., [1,2]]
        x = x.contiguous().permute(0,1,4,2,3).reshape(b, -1, h//2, w//2)
        y = y.contiguous().permute(0,1,4,2,3).reshape(b, -1, h//2, w//2)
        return x, y
    @staticmethod
    def merge(x: torch.Tensor, y: torch.Tensor):
        b, _, h2, w2 = x.shape
        x = x.view(b, -1, 2, h2, w2) #index [0,3]
        y = y.view(b, -1, 2, h2, w2) #index [1,2]
        x1, x2 = torch.chunk(x, 2, dim=2)
        out = torch.cat([x1, y, x2], dim=2).contiguous().permute(0,1,3,4,2)
        out = out.view(b, -1, h2, w2, 2, 2).permute(0,1,2,4,3,5).reshape(b, -1, h2*2, w2*2)
        return out

# class SpatialMasking:
#     weights = torch.tensor([[[1,0], [0,0]],
#                                  [[0,0], [0,1]],
#                                  [[0,1], [0,0]],
#                                  [[0,0], [1,0]]], dtype=torch.float32).view(4,1,2,2)

#     @staticmethod
#     def split(x: torch.Tensor):
#         b, c, h, w = x.shape
#         weight = SpatialMasking.weights.repeat(1,c,1,1).view(-1,1,2,2)
#         x = F.conv2d(x, weight, groups=c)
#         x, y = torch.chunk(x,2, dim = 1)
#         return x, y
#     @staticmethod
#     def merge(x: torch.Tensor, y: torch.Tensor):
#         x = torch.cat([x,y], dim = 1)
#         b, c, h2, w2 = x.shape
#         weight = SpatialMasking.weights.repeat(1,c//4,1,1).view(-1,1,2,2)

#         out = F.conv_transpose2d(x, weight, groups=c//4)
#         return out

class NestedAffineCoupling(nn.Module):
    eps = 1e-10
    def __init__(self, in_channels, condition_channels = None, level = 2, masking_type = 'auto', max_stack_level = 2, enforce_channel_on_base = True, name = "top", *args, **kwargs) -> None:
        '''
        :param: in_channels (int): The number of input channels.
        :param: condition_channels (Optinal[int]): The number of conditional channels. Default: None.
        :param: masking_type (Optinal[str]): The type of masking used at each level.\n
            - "channel": mask/split the feature along the channel dimension.
            - "spatial": mask/split the feature along the spatial dimension.
            - "alter": Alternate between "channel" and "spatial", with "channel" if level is even.
            - "reverse_alter": Alternate between "channel" and "spatial", with "channel" if level is odd.
            - "auto" (default): Choose the "alter" option if the start level is even and otherwise.\n
            Only choose "channel"/"spatial" may lead to vanishing/exploding number of parameters (as the channel/spatial masking half/double the number of the channel).\n
            This option will be ignored if a custom masking type of the current level is provied (e.g. `custom_masking = {level: masking_type, ...}`)
        :param: max_stack_level (Optinal[int]): The maximum number of flow blocks used at each level. Default: 2.\n
            Becaution that this make the number of flow blocks grow exponentially, so this is advised to be untouched.\n
            By default, to stop this behaviour, only the first level will follow this number, all the lower level will be\n
            recalculated as max_stack_level = min(max_stack_level, level). To disable this behaviour, set `constrain_stack = False`.
        :param: enforce_channel_on_base (Optinal[bool]): Decide if the base flow (where there won't be any lower flow) will always use channel masking. Default: True.\n
        :param: custom_masking (Optinal[dict]): A dictionary with level as key and masking_type as value. Masking type must be "channel" or "spatial". Default: None \n
            If the current level is not provided by this dict, the parameter masking_type will be used.
        '''
        super().__init__()
        self.level = level
        self.name = name
        self.is_first_level = kwargs.get('is_first_level', True)
        kwargs.pop('is_first_level', None) 
        if not self.is_first_level and kwargs.get('constrain_stack', True):
            self.stack_level = min(max(max_stack_level,1), max(level,1))
        else:
            self.stack_level = max(max_stack_level,1)
        


        custom_mask = kwargs.get("custom_masking", None)
        if custom_mask is not None:
            assert isinstance(custom_mask, dict)
            if level in custom_mask:
                _mask_type = custom_mask.get(level)
                assert _mask_type in ["channel", "spatial"]
                self.masking_type = _mask_type

        if custom_mask is None or level not in custom_mask:
            if masking_type == 'auto':
                if self.is_first_level:
                    masking_type = 'alter' if level % 2 == 0 else 'reverse_alter'
                else:
                    masking_type = 'alter'

            if masking_type == 'alter':
                self.masking_type = 'channel' if level % 2 == 0 else 'spatial'
            elif masking_type == 'reverse_alter':
                self.masking_type = 'channel' if level % 2 != 0 else 'spatial'
            else:
                self.masking_type = masking_type

        if self.masking_type == 'channel':
            if in_channels == 1:
                warnings.warn("Trying to use Channel masking on a 1-channel feature map is not available. Setting to use Spatial masking instead.")
                self.masking_type = 'spatial'
                self.masking_func = SpatialMasking
            else:
                self.masking_func = ChannelMasking
        elif self.masking_type == 'spatial':
            self.masking_func = SpatialMasking
        else:
            raise NotImplementedError
        
        if enforce_channel_on_base and level == 0:
            self.masking_func = ChannelMasking
            self.masking_type = 'channel'
    
        if level == 0:
            if self.masking_type == 'channel':
                in_channels_ = in_channels // 2
            elif self.masking_type == 'spatial':
                in_channels_ = in_channels * 2
            self.stack_level = 1 # set this explicitly to constrain the rate of parameters 
            
            self.net1 = nn.ModuleList([
                    NN(in_channels_, condition_channels, in_channels_, in_channels_ * 2, False) for _ in range(self.stack_level)])
            self.net2 = nn.ModuleList([
                    NN(in_channels_, condition_channels, in_channels_, in_channels_ * 2, False) for _ in range(self.stack_level)])
        else:
            sample_level = torch.linspace(1, self.level, self.stack_level).round().int().tolist()
            if self.masking_type == 'channel':
                self.flow1 = nn.ModuleList([
                    NestedAffineCoupling(in_channels//2, in_channels//2, 
                                         level - i, masking_type, max_stack_level = max_stack_level, is_first_level = False, enforce_channel_on_base= enforce_channel_on_base,
                                         name = f'{name}_flow1_{i}', *args, **kwargs) for i in sample_level])
                self.flow2 = nn.ModuleList([
                    NestedAffineCoupling(in_channels//2, in_channels//2, 
                                         level - i, masking_type, max_stack_level = max_stack_level, is_first_level = False, enforce_channel_on_base= enforce_channel_on_base, 
                                         name = f'{name}_flow2_{i}',*args, **kwargs) for i in sample_level])
                self.condition_module = nn.ModuleList([ConditionalModule(in_channels//2, condition_channels) for i in sample_level])
            elif self.masking_type == 'spatial':
                self.flow1 = nn.ModuleList([
                    NestedAffineCoupling(in_channels*2, in_channels*2, 
                                         level - i, masking_type, max_stack_level = max_stack_level, is_first_level = False, enforce_channel_on_base= enforce_channel_on_base, 
                                         name = f'{name}_flow1_{i}', *args, **kwargs) for i in sample_level])
                self.flow2 = nn.ModuleList([
                    NestedAffineCoupling(in_channels*2, in_channels*2, 
                                         level - i, masking_type, max_stack_level = max_stack_level, is_first_level = False, enforce_channel_on_base= enforce_channel_on_base, 
                                         name = f'{name}_flow2_{i}',*args, **kwargs) for i in sample_level])
                self.condition_module = nn.ModuleList([ConditionalModule(in_channels*2, condition_channels) for i in sample_level])

        # self.perm = nn.ModuleList([
        #             InvertibleSequential(InvConv2dLU(in_channels),
        #                                  InvertibleLeakyReLU(),
        #                                  InvConv2dLU(in_channels)) for _ in range(self.stack_level)])
        
        # self.norm = nn.ModuleList([
        #             ActNorm(in_channels) for _ in range(self.stack_level)])
        
        self.norm = nn.ModuleList([
                    Identity(in_channels) for _ in range(self.stack_level)])
        
        self.perm = nn.ModuleList([
                    InvertibleSequential(Identity(in_channels),
                                         InvertibleLeakyReLU(),
                                         Identity(in_channels)) for _ in range(self.stack_level)])
 

    def f_clamp(self, x: torch.Tensor):
        # return x/2.
        return 0.636 * torch.atan(x)
    
    def forward(self, x: torch.Tensor, x_cond: Optional[torch.Tensor] = None):
        # print(self.level)
        if self.level == 0:
            x_change, x_id = self.masking_func.split(x)
            for i in range(self.stack_level):
                x_ = self.masking_func.merge(x_change, x_id)
                x_ = self.norm[i](x_)
                x_ = self.perm[i](x_)
                
                x_change, x_id = self.masking_func.split(x_)
                if x_cond is not None:
                    st: torch.Tensor = self.net1[i](x_id, x_cond)
                else:
                    st: torch.Tensor = self.net1[i](x_id)
                
                s, t = st.chunk(2, 1)
                s = 2. * self.f_clamp(s)
                # print(self.name, s.mean())
                x_change = x_change * torch.exp(s) + t

                x_change2, x_id2 = x_id, x_change
                if x_cond is not None:
                    st2: torch.Tensor = self.net2[i](x_id2, x_cond)
                else:
                    st2: torch.Tensor = self.net2[i](x_id2)
                s2, t2 = st2.chunk(2, 1)
                s2 = 2. * self.f_clamp(s2)
                # print(self.name, s2.mean())
                x_change2 = x_change2 * torch.exp(s2) + t2
                
                x_change, x_id = x_id2, x_change2

            return self.masking_func.merge(x_change, x_change2)
        
        else:
            x_change, x_id = self.masking_func.split(x)
            for i in range(self.stack_level):
                x_ = self.masking_func.merge(x_change, x_id)
                x_ = self.norm[i](x_)
                x_ = self.perm[i](x_)
                x_change, x_id = self.masking_func.split(x_)
                if x_cond is not None:
                    x_cond1 = self.condition_module[i](x_id, x_cond)
                    x_change = self.flow1[i](x_change, x_cond1)
                else:
                    x_change = self.flow1[i](x_change, x_id)
                x_change2, x_id2 = x_id, x_change
                if x_cond is not None:
                    x_cond2 = self.condition_module[i](x_id2, x_cond)
                    x_change2 = self.flow2[i](x_change2, x_cond2)
                else:
                    x_change2 = self.flow2[i](x_change2, x_id2)

                # print(x_change.shape, x_id.shape, x_cond.shape, i)

                # x_id2, x_change2 = self.masking_func.split(self.perm[i](self.masking_func.merge(x_id2, x_change2)))
                x_change, x_id = x_id2, x_change2

            return self.masking_func.merge(x_id2, x_change2)
        
    def inverse(self, x: torch.Tensor, x_cond: Optional[torch.Tensor] = None):
        if self.level == 0:
            x_id, x_change = self.masking_func.split(x)
            for i in reversed(range(self.stack_level)):
                # x_id, x_change = self.masking_func.split(self.perm[i].inverse(self.masking_func.merge(x_id, x_change)))
                if x_cond is not None:
                    st: torch.Tensor = self.net2[i](x_id, x_cond)
                else:
                    st: torch.Tensor = self.net2[i](x_id)
                s, t = st.chunk(2, 1)
                s = 2. * self.f_clamp(s)
                x_change = (x_change - t) / torch.exp(s)
                # print(self.name, s.mean())
                x_id2, x_change2 = x_change, x_id
                if x_cond is not None:
                    st2: torch.Tensor = self.net1[i](x_id2, x_cond)
                else:
                    st2: torch.Tensor = self.net1[i](x_id2)
                s2, t2 = st2.chunk(2, 1)
                s2 = 2. * self.f_clamp(s2)
                # print(self.name, s2.mean())
                x_change2 = (x_change2 - t2) / torch.exp(s2)
                x_ = self.masking_func.merge(x_change2, x_id2)
                x_ = self.perm[i].inverse(x_)
                x_ = self.norm[i].inverse(x_)
                x_change2, x_id2 = self.masking_func.split(x_)
                x_id, x_change = x_change2, x_id2

            return self.masking_func.merge(x_change2, x_id2)
        
        else:
            x_id2, x_change2 = self.masking_func.split(x)
            for i in reversed(range(self.stack_level)):
                # print(x_change2.shape, x_id2.shape, x_cond.shape, i)
                x_id2, x_change2 = self.masking_func.split(self.perm[i].inverse(self.masking_func.merge(x_id2, x_change2)))
                if x_cond is not None:
                    x_cond2 = self.condition_module[i](x_id2, x_cond)
                    x_change2 = self.flow2[i].inverse(x_change2, x_cond2)
                else:
                    x_change2 = self.flow2[i].inverse(x_change2, x_id2)
                x_id, x_change = x_change2, x_id2
                if x_cond is not None:
                    x_cond1 = self.condition_module[i](x_id, x_cond)
                    x_change = self.flow1[i].inverse(x_change, x_cond1)
                else:
                    x_change = self.flow1[i].inverse(x_change, x_id)

                x_ = self.masking_func.merge(x_change2, x_id2)
                x_ = self.perm[i].inverse(x_)
                x_ = self.norm[i].inverse(x_)
                x_change, x_id = self.masking_func.split(x_)
                x_id2, x_change2 =  x_change, x_id

            return self.masking_func.merge(x_change, x_id)
    # def forward(self, x: torch.Tensor, x_cond: Optional[torch.Tensor] = None):
    #     # print(self.level)
    #     if self.level == 0:
    #         for i in range(self.stack_level):
                
    #             x_change, x_id = self.masking_func.split(x)
    #             if x_cond is not None:
    #                 st: torch.Tensor = self.net1[i](x_id, x_cond)
    #             else:
    #                 st: torch.Tensor = self.net1[i](x_id)
                
    #             s, t = st.chunk(2, 1)
    #             s = 2. * self.f_clamp(s)
    #             # print(self.name, s.mean())
    #             x_change = x_change * torch.exp(s) + t

    #             x_change2, x_id2 = x_id, x_change
    #             if x_cond is not None:
    #                 st2: torch.Tensor = self.net2[i](x_id2, x_cond)
    #             else:
    #                 st2: torch.Tensor = self.net2[i](x_id2)
    #             s2, t2 = st2.chunk(2, 1)
    #             s2 = 2. * self.f_clamp(s2)
    #             # print(self.name, s2.mean())
    #             x_change2 = x_change2 * torch.exp(s2) + t2
                
    #             x_change, x_id = x_id2, x_change2
    #             x = self.masking_func.merge(x_change, x_id)
    #             x = self.norm[i](x)
    #             x = self.perm[i](x)

    #         return x
        
    #     else:
    #         for i in range(self.stack_level):
                
    #             x_change, x_id = self.masking_func.split(x)
    #             if x_cond is not None:
    #                 x_cond1 = self.condition_module[i](x_id, x_cond)
    #                 x_change = self.flow1[i](x_change, x_cond1)
    #             else:
    #                 x_change = self.flow1[i](x_change, x_id)
    #             x_change2, x_id2 = x_id, x_change
    #             if x_cond is not None:
    #                 x_cond2 = self.condition_module[i](x_id2, x_cond)
    #                 x_change2 = self.flow2[i](x_change2, x_cond2)
    #             else:
    #                 x_change2 = self.flow2[i](x_change2, x_id2)

    #             # print(x_change.shape, x_id.shape, x_cond.shape, i)

    #             x_change, x_id = x_id2, x_change2
    #             x = self.masking_func.merge(x_change, x_id)
    #             x = self.norm[i](x)
    #             x = self.perm[i](x)    

    #         return x
        
    # def inverse(self, x: torch.Tensor, x_cond: Optional[torch.Tensor] = None):
    #     if self.level == 0:
    #         for i in reversed(range(self.stack_level)):
    #             x = self.perm[i].inverse(x)
    #             x = self.norm[i].inverse(x)
    #             x_id, x_change = self.masking_func.split(x)
    #             if x_cond is not None:
    #                 st: torch.Tensor = self.net2[i](x_id, x_cond)
    #             else:
    #                 st: torch.Tensor = self.net2[i](x_id)
    #             s, t = st.chunk(2, 1)
    #             s = 2. * self.f_clamp(s)
    #             x_change = (x_change - t) / torch.exp(s)
    #             # print(self.name, s.mean())
    #             x_id2, x_change2 = x_change, x_id
    #             if x_cond is not None:
    #                 st2: torch.Tensor = self.net1[i](x_id2, x_cond)
    #             else:
    #                 st2: torch.Tensor = self.net1[i](x_id2)
    #             s2, t2 = st2.chunk(2, 1)
    #             s2 = 2. * self.f_clamp(s2)
    #             # print(self.name, s2.mean())
    #             x_change2 = (x_change2 - t2) / torch.exp(s2)

    #             x_id, x_change = x_change2, x_id2
    #             x = self.masking_func.merge(x_id, x_change)
                

    #         return x
        
    #     else:
            
    #         for i in reversed(range(self.stack_level)):
    #             # print(x_change2.shape, x_id2.shape, x_cond.shape, i)
    #             x = self.perm[i].inverse(x)
    #             x = self.norm[i].inverse(x)
    #             x_id2, x_change2 = self.masking_func.split(x)
    #             if x_cond is not None:
    #                 x_cond2 = self.condition_module[i](x_id2, x_cond)
    #                 x_change2 = self.flow2[i].inverse(x_change2, x_cond2)
    #             else:
    #                 x_change2 = self.flow2[i].inverse(x_change2, x_id2)
    #             x_id, x_change = x_change2, x_id2
    #             if x_cond is not None:
    #                 x_cond1 = self.condition_module[i](x_id, x_cond)
    #                 x_change = self.flow1[i].inverse(x_change, x_cond1)
    #             else:
    #                 x_change = self.flow1[i].inverse(x_change, x_id)

    #             x_id2, x_change2 =  x_change, x_id
    #             x = self.masking_func.merge(x_id, x_change)
                

    #         return x
        
    def check_masking_type(self):
        def inner_check(module: NestedAffineCoupling, check_level = []):
            if module.level not in check_level:
                print(module.level, module.masking_type)
                check_level.append(module.level)
                if module.level != 0:
                    for mod in module.flow1:
                        inner_check(mod, check_level)
        inner_check(self, [])
                
    def check_stack_level(self):
        def inner_check(module: NestedAffineCoupling, check_level = []):
            if module.level not in check_level:
                print(module.level, module.stack_level)
                check_level.append(module.level)
                if module.level != 0:
                    for mod in module.flow1:
                        inner_check(mod, check_level)
        inner_check(self, [])
        
    def check_total_flow(self, verbose = True):
        n_flow = 0
        def inner_check(module: NestedAffineCoupling):
            nonlocal n_flow
            n_flow += 1
            if module.level != 0:
                for mod in module.flow1:
                    inner_check(mod)
                for mod in module.flow2:
                    inner_check(mod)
                # We dont need to count flow2 as a coupling block should contain 2 "transformation"
        inner_check(self)
        if verbose:
            print(n_flow)
        return n_flow

class NN(nn.Module):
    """Small convolutional network used to compute scale and translate factors.

    Args:
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the hidden activations.
        out_channels (int): Number of channels in the output.
        use_act_norm (bool): Use activation norm rather than batch norm.
    """
    def __init__(self, in_channels, cond_channels, mid_channels, out_channels,
                 use_half_instance=True):
        super(NN, self).__init__()
        norm_fn = LocalHalfInstanceNorm2d if use_half_instance else nn.InstanceNorm2d
        if cond_channels is None:
            cond_channels = 0

        self.in_norm = norm_fn(in_channels)
        self.in_conv = nn.Conv2d(in_channels + cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_conv1 = nn.Conv2d(mid_channels + cond_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.mid_conv1.weight, 0., 0.05)

        self.mid_norm = norm_fn(mid_channels)
        self.mid_conv2 = nn.Conv2d(mid_channels + cond_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)

        self.out_norm = norm_fn(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        
        nn.init.normal_(self.mid_conv2.weight, 0., 0.05)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x: torch.Tensor, x_cond: Optional[torch.Tensor] = None):
        if x_cond is not None:
            x_cond = F.interpolate(x_cond, x.shape[-2:])
        x = self.in_norm(x)
        x = self.in_conv(torch.cat([x, x_cond], dim = 1) if x_cond is not None else x)
        x = F.relu(x)

        x = self.mid_conv1(torch.cat([x, x_cond], dim = 1) if x_cond is not None else x)
        x = self.mid_norm(x)
        x = F.relu(x)

        x = self.mid_conv2(torch.cat([x, x_cond], dim = 1) if x_cond is not None else x)
        x = self.out_norm(x)
        x = F.relu(x)

        x = self.out_conv(x)

        return x
    

class ConditionalModule(nn.Module):
    def __init__(self, in_channels, cond_channels):
        super().__init__()
        if cond_channels is None:
            cond_channels = 0
        self.total_in_channels = in_channels + cond_channels
        self.in_norm = nn.InstanceNorm2d(self.total_in_channels)
        self.pw_conv = nn.Conv2d(self.total_in_channels, in_channels, kernel_size=1, padding=0, bias=True)
        self.dw_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False, groups= in_channels)

    def forward(self, x: torch.Tensor, x_cond: torch.Tensor):
        x_cond = F.interpolate(x_cond, x.shape[-2:])
        x = self.in_norm(torch.cat([x, x_cond], dim = 1))
        x = F.relu(self.pw_conv(x))
        x = self.dw_conv(x)
        return x
    
class MaskedSpatialMasking:
    @staticmethod
    def split(x: torch.Tensor):
        b, c, h, w = x.shape
        mask = torch.zeros(1,1,h,w, dtype= x.dtype, device= x.device)
        mask[..., ::2, ::2] = 1.
        mask[..., 1::2, 1::2] = 1.
        y = x*(1-mask)
        x = x*mask
        return x, y
    @staticmethod
    def merge(x: torch.Tensor, y: torch.Tensor):
        b, c, h, w = x.shape
        mask = torch.zeros(1,1,h,w, dtype= x.dtype, device= x.device)
        mask[..., ::2, ::2] = 1.
        mask[..., 1::2, 1::2] = 1.
        y = x*(1-mask)
        x = x*mask
        out = x+y
        return out

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, masking_type = "channel", *args, **kwargs):
        super().__init__()
        if masking_type == 'channel':
            self.masking_func = ChannelMasking
            self.masking_type = 'channel'
        elif masking_type == 'spatial':
            self.masking_func = MaskedSpatialMasking
            self.masking_type = 'spatial'
        else:
            raise NotImplementedError
        
        if self.masking_type == 'channel':
                in_channels_ = in_channels // 2
        elif self.masking_type == 'spatial':
            in_channels_ = in_channels
        self.net1 = NN(in_channels_, 0, in_channels_, in_channels_ * 2, False)
        self.net2 = NN(in_channels_, 0, in_channels_, in_channels_ * 2, False)
        self.perm = InvertibleSequential(InvConv2dLU(in_channels),
                                         InvertibleLeakyReLU(),
                                         InvConv2dLU(in_channels))

    def f_clamp(self, x: torch.Tensor):
        return 0.636 * torch.atan(x)

    def forward(self, x: torch.Tensor):
        # print(self.level)
        x_change, x_id = self.masking_func.split(x)
        # if x_cond is not None:
        #     st: torch.Tensor = self.net1(x_id, x_cond)
        # else:
        st: torch.Tensor = self.net1(x_id)
        
        s, t = st.chunk(2, 1)
        s = 2. * self.f_clamp(s)
        x_change = x_change * torch.exp(s) + t

        x_change2, x_id2 = x_id, x_change
        # if x_cond is not None:
        #     st2: torch.Tensor = self.net2(x_id2, x_cond)
        # else:
        st2: torch.Tensor = self.net2(x_id2)
        s2, t2 = st2.chunk(2, 1)
        s2 = 2. * self.f_clamp(s2)
        x_change2 = x_change2 * torch.exp(s2) + t2

        return self.perm(self.masking_func.merge(x_change, x_change2))

        
    def inverse(self, x: torch.Tensor):
        x = self.perm.inverse(x)
        x_id, x_change = self.masking_func.split(x)
        # if x_cond is not None:
        #     st: torch.Tensor = self.net2(x_id, x_cond)
        # else:
        st: torch.Tensor = self.net2(x_id)
        s, t = st.chunk(2, 1)
        s = 2. * self.f_clamp(s)
        x_change = (x_change - t) * torch.exp(-s)

        x_id2, x_change2 = x_change, x_id
        # if x_cond is not None:
        #     st2: torch.Tensor = self.net1(x_id2, x_cond)
        # else:
        st2: torch.Tensor = self.net1(x_id2)
        s2, t2 = st2.chunk(2, 1)
        s2 = 2. * self.f_clamp(s2)
        x_change2 = (x_change2 - t2) * torch.exp(-s2)

        return self.masking_func.merge(x_change2, x_change)
        

class InvConv2d(nn.Module):
    def __init__(self, in_channel, out_channel=None):
        super().__init__()
        self.in_channel = in_channel
        if out_channel is None:
            out_channel = in_channel
        weight = torch.randn(in_channel, out_channel)
        q, _ = torch.linalg.qr(weight, mode= 'complete')
        q: torch.Tensor
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input: torch.Tensor) -> "torch.Tensor":
        out = F.conv2d(input, self.weight)
        return out

    def inverse(self, output: torch.Tensor) -> "torch.Tensor":
        return F.conv2d(
            output, self.weight.squeeze(3).squeeze(2).inverse().unsqueeze(2).unsqueeze(3)
        )
    

class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        w_p, w_l, w_u = torch.linalg.lu(q.float())
        w_s = torch.diag(w_u)
        w_u = torch.triu(w_u, 1)
        u_mask = torch.triu(torch.ones_like(w_u), 1)
        l_mask = u_mask.T

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", u_mask)
        self.register_buffer("l_mask", l_mask)
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(torch.log(torch.abs((w_s))))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        return out

    def calc_weight(self):
        weight: torch.Tensor = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )

        return weight.unsqueeze(2).unsqueeze(3)

    def inverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class InvertibleLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__(negative_slope, inplace)

    def inverse(self, x):
        return F.leaky_relu(x, negative_slope= 1/self.negative_slope, inplace=self.inplace)




class InvertibleSequential(nn.Sequential):

    def inverse(self, x):
        for block in reversed(self):
            x = block.inverse(x)

        return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x 
    def inverse(self, x):
        return x
    

class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.in_channel = in_channel
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
        self.initialized: torch.Tensor

    def initialize(self, input: torch.Tensor):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input: torch.Tensor):
        _, _, height, width = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
            
        return self.scale * (input + self.loc)

    def inverse(self, output: torch.Tensor):
        return output / self.scale - self.loc