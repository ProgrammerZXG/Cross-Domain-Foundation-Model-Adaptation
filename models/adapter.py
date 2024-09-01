import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.vision_transformer_lora import vit_small_lora,vit_base_lora
from models.vision_transformer import vit_small,vit_base
import fvcore.nn.weight_init as weight_init

import logging
import loralib as lora

_DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    logger = logging.getLogger("dinov2")
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    if checkpoint_key is not None and checkpoint_key in state_dict:
        logger.info(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def make_dinov2_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov2_{compact_arch_name}{patch_size}"

def make_vit_encoder(dino_pretrain="False",vit_type="small",finetune_method="unfrozen"):
    vit_kwargs = dict(
        in_chans = 3,
        img_size=224,
        patch_size=14,
        init_values=1.0e-05,
        ffn_layer="mlp",
        block_chunks=0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True
    )
    if dino_pretrain == "True":
        model_name = make_dinov2_model_name("vit_"+vit_type, 14)
        url = _DINOV2_BASE_URL + f"/{model_name}/{model_name}_pretrain.pth"
        pretrained_weights = torch.hub.load_state_dict_from_url(url, map_location="cpu")
    if finetune_method == "unfrozen" or finetune_method == "frozen":
        if vit_type == "small":
            encoder = vit_small(**vit_kwargs)
            emb = 384
            strict = True
        elif vit_type == "base":
            encoder = vit_base(**vit_kwargs)
            emb = 768
            strict = True
        else:
            print("Error in vit_type!!!")
    elif finetune_method == "lora":
        if vit_type == "small":
            encoder = vit_small_lora(**vit_kwargs)
            emb = 384
            strict = False
        elif vit_type == "base":
            encoder = vit_base_lora(**vit_kwargs)
            emb = 768
            strict = False
        else:
            print("Error in vit_type!!!")
    if dino_pretrain == "True":
        encoder.load_state_dict(pretrained_weights, strict=strict)
    return encoder,emb

class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs


class SETR_PUP(nn.Module):
    def __init__(self,embedding_dim,num_classes):
        super(SETR_PUP,self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        extra_in_channels = int(self.embedding_dim/4)
        in_channels = [
            self.embedding_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, out_channels)
        ):
            modules.append(
                self.conv_block(in_channel,out_channel)
            )
            modules.append(nn.Upsample(size=(1//(2**(3-i)),1//(2**(3-i))), mode='bilinear'))
        
        modules.append(
            nn.Conv2d(
                in_channels=out_channels[-1], out_channels=self.num_classes,
                kernel_size=1, stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            ))
        self.decode_net = IntermediateSequential(
            *modules, return_intermediate=False
        )

    def forward(self,x,size):
        n1,n2 = size
        self.decode_net[1] = nn.Upsample(size=(n1//(2**(3)),n2//(2**(3))), mode='bilinear')
        self.decode_net[3] = nn.Upsample(size=(n1//(2**(2)),n2//(2**(2))), mode='bilinear')
        self.decode_net[5] = nn.Upsample(size=(n1//(2**(1)),n2//(2**(1))), mode='bilinear')
        self.decode_net[7] = nn.Upsample(size=(n1,n2), mode='bilinear')
        return self.decode_net(x)

    def conv_block(self,in_channels, out_channels):
        conv = nn.Sequential(
                nn.Conv2d(
                    int(in_channels), int(out_channels), 3, 1,
                    padding=self._get_padding('SAME', (3, 3),),
                ),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(
                    int(out_channels), int(out_channels), 3, 1,
                    padding=self._get_padding('SAME', (3, 3),),
                ),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True)
                )
        return conv

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

class SETR_MLA(nn.Module):
    def __init__(self,embedding_dim,num_classes):
        super(SETR_MLA,self).__init__()

        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.net1_in, self.net1_intmd, self.net1_out = self._define_agg_net()
        self.net2_in, self.net2_intmd, self.net2_out = self._define_agg_net()
        self.net3_in, self.net3_intmd, self.net3_out = self._define_agg_net()
        self.net4_in, self.net4_intmd, self.net4_out = self._define_agg_net()

        self.output_net = IntermediateSequential(return_intermediate=False)
        self.output_net.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=self.embedding_dim, out_channels=self.num_classes,
                kernel_size=1, stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            )
        )
        self.output_net.add_module(
            "upsample_1",
            nn.Upsample(size = (1,1), mode='bilinear')
        )

    def forward(self,x,size):
        n1,n2 = size
        self.output_net[-1] = nn.Upsample(size = (n1,n2), mode='bilinear')
        x3,x6,x9,x12 = x
        
        x12_intmd_in = self.net1_in(x12)
        x12_out = self.net1_out(x12_intmd_in)

        x9_in = self.net2_in(x9)
        x9_intmd_in = x9_in + x12_intmd_in
        x9_intmd_out = self.net2_intmd(x9_intmd_in)
        x9_out = self.net2_out(x9_intmd_out)

        x6_in = self.net3_in(x6)
        x6_intmd_in = x6_in + x9_intmd_in
        x6_intmd_out = self.net3_intmd(x6_intmd_in)
        x6_out = self.net3_out(x6_intmd_out)

        x3_in = self.net4_in(x3)
        x3_intmd_in = x3_in + x6_intmd_in
        x3_intmd_out = self.net4_intmd(x3_intmd_in)
        x3_out = self.net4_out(x3_intmd_out)

        out = torch.cat((x12_out, x9_out, x6_out, x3_out), dim=1)
        out = self.output_net(out) 

        return out

    def conv_block(self,in_channels, out_channels):
        conv = nn.Sequential(
                nn.Conv2d(
                    int(in_channels), int(out_channels), 3, 1,
                    padding=self._get_padding('SAME', (3, 3),),
                ),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(
                    int(out_channels), int(out_channels), 3, 1,
                    padding=self._get_padding('SAME', (3, 3),),
                ),
                nn.BatchNorm2d(int(out_channels)),
                nn.ReLU(inplace=True)
                )
        return conv
    
    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            self.conv_block(self.embedding_dim,int(self.embedding_dim/2))
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            self.conv_block(int(self.embedding_dim/2),int(self.embedding_dim/2))
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            self.conv_block(int(self.embedding_dim/2),int(self.embedding_dim/2))
        )
        model_out.add_module(
            "layer_3",
            self.conv_block(int(self.embedding_dim/2),int(self.embedding_dim/4))
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode='bilinear')
        )
        model_out.add_module(
            "layer_4",
            self.conv_block(int(self.embedding_dim/4),int(self.embedding_dim/4))
        )
        return model_in, model_intmd, model_out

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

class dinov2_pup(nn.Module):
    def __init__(self, num_classes, pretrain = True, vit_type="small",frozen=False,finetune_method="unfrozen"):
        super(dinov2_pup,self).__init__()

        self.encoder, self.emb = make_vit_encoder(pretrain,vit_type,finetune_method)
        self.decoder = SETR_PUP(self.emb, num_classes)

        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            if finetune_method == "unfrozen":
                for param in self.encoder.parameters():
                    param.requires_grad = True 
            elif finetune_method == "lora":  
                lora.mark_only_lora_as_trainable(self.encoder)  
 
    def forward(self,x,size):
        B,_,H,W =  x.shape
        features,_ = self.encoder.forward_features(x)
        fea_img = features['x_norm_patchtokens']
        fea_img = fea_img.view(fea_img.size(0),int(H / 14),int(W / 14),self.emb)
        fea_img = fea_img.permute(0, 3, 1, 2).contiguous()
        out = self.decoder(fea_img,size)
        return out
        
class dinov2_mla(nn.Module):
    def __init__(self, num_classes, pretrain = True, vit_type="small",frozen=False,finetune_method="unfrozen"):
        super(dinov2_mla,self).__init__()

        self.encoder, self.emb = make_vit_encoder(pretrain,vit_type,finetune_method)
        self.decoder = SETR_MLA(self.emb, num_classes)
        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            if finetune_method == "unfrozen":
                for param in self.encoder.parameters():
                    param.requires_grad = True 
            elif finetune_method == "lora":  
                lora.mark_only_lora_as_trainable(self.encoder)  

    def forward(self,x,size):
        B,_,H,W =  x.shape
        _, x_middle = self.encoder.forward_features(x)
        xm = []
        for k,x in x_middle.items():
            x = x.view(
                x.size(0),
                int(H / 14),
                int(W / 14),
                self.emb,
            )
            x = x.permute(0, 3, 1, 2).contiguous()
            xm.append(x)
        out = self.decoder(xm,size)
        return out
        
class dinov2_linear(nn.Module):
    def __init__(self, num_classes, pretrain = True, vit_type="small",frozen=False,finetune_method="unfrozen"):
        super(dinov2_linear,self).__init__()

        self.encoder, self.emb = make_vit_encoder(pretrain,vit_type,finetune_method)
        self.decoder = nn.Conv2d(self.emb, num_classes, kernel_size=1)

        if frozen:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            if finetune_method == "unfrozen":
                for param in self.encoder.parameters():
                    param.requires_grad = True 
            elif finetune_method == "lora":  
                lora.mark_only_lora_as_trainable(self.encoder)  
 
    def forward(self,x,size):
        B,_,H,W =  x.shape
        features,_ = self.encoder.forward_features(x)
        fea_img = features['x_norm_patchtokens']
        fea_img = fea_img.view(fea_img.size(0),int(H / 14),int(W / 14),self.emb)
        fea_img = fea_img.permute(0, 3, 1, 2).contiguous()
        out = self.decoder(fea_img)
        out = F.interpolate(out,size=size)
        return out

    