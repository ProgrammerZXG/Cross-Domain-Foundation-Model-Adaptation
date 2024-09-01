import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vision_transformer_lora import vit_small_lora,vit_base_lora
from models.vision_transformer import vit_small,vit_base
from models.dpt import _make_fusion_block,_make_scratch
import logging
import loralib as lora
########################################################################################################################

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

class dinov2_dpt(nn.Module):
    def __init__(self, num_classes, pretrain = True, vit_type="small",frozen=False,finetune_method="unfrozen"):
        super(dinov2_dpt,self).__init__()

        features = 256

        self.encoder, self.emb = make_vit_encoder(pretrain,vit_type,finetune_method)
        self.scratch = _make_scratch([self.emb,self.emb,self.emb,self.emb],
                                     out_shape=features)
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn=True)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn=True)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn=True)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn=True)
        
        self.scratch.single_conv = nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1)
        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

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
        layer_1, layer_2, layer_3, layer_4 = xm
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4((size[0]//16,size[1]//16), layer_4_rn)
        path_3 = self.scratch.refinenet3((size[0]//8,size[1]//8),path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2((size[0]//4,size[1]//4),path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1((size[0]//2,size[1]//2),path_2, layer_1_rn)

        out = self.scratch.single_conv(path_1)
        out = F.interpolate(out,size=size)
        out = self.scratch.output_conv(out)
        return out

if __name__ == "__main__":
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    model = dinov2_dpt(1).to(device=device)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model Total: %d'%total_num)
    print('Model Trainable: %d'%trainable_num)
    x1 = torch.Tensor(1,3,434,994).to(device=device,dtype=torch.float32)
    y1 = model(x1,size=(434,994))
    print(x1.shape)
    print(y1.shape)