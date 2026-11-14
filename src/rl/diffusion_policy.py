"""Image-conditioned DDPO-SF policy; source/text/guidance mirror IP2Pix."""
import copy
import time
import types
import torch
from ..training.generator_adapter import add_attention_lora, adapter_state
from .diffusion_trajectory import transition

INSTRUCTION = "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged."

def fp32_adapter_linear(module,inputs):
    # PEFT still combines/scales the adapters in its normal forward. Only small
    # A/B linear operators escape autocast; frozen base computation remains BF16.
    with torch.autocast(device_type=inputs.device.type,enabled=False):
        return torch.nn.functional.linear(inputs.float(),module.weight,module.bias)

def cfg_inputs(latent, image_latent):
    images = torch.cat([image_latent, image_latent, torch.zeros_like(image_latent)])
    return torch.cat([torch.cat([latent]*3), images.float()],dim=1)

def guided_prediction(prediction, text_scale, image_scale):
    text, image, unconditional = prediction.chunk(3)
    return unconditional + text_scale*(text-image) + image_scale*(image-unconditional)

class DiffusionPolicy:
    def __init__(self,editor,config):
        from diffusers import DDIMScheduler
        self.editor,self.pipe,self.config=editor,editor.pipe,dict(config)
        self.parameters=add_attention_lora(self.pipe,config["lora_rank"],config["lora_alpha"])
        if config.get("adapter_compute_precision")=="fp32":
            torch.backends.cuda.matmul.allow_tf32=False
            for name,module in self.pipe.unet.named_modules():
                if isinstance(module,torch.nn.Linear) and (".lora_A." in name or ".lora_B." in name):
                    module.forward=types.MethodType(fp32_adapter_linear,module)
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        self.pipe.scheduler=DDIMScheduler.from_config(self.pipe.scheduler.config,
            set_alpha_to_one=False,steps_offset=1,clip_sample=False,timestep_spacing="leading")
        self.pipe.scheduler.set_timesteps(config["num_inference_steps"],device=editor.device)
        self.pipe.unet.train()
        if any(isinstance(m,torch.nn.Dropout) and m.p > 0 for m in self.pipe.unet.modules()):
            raise ValueError("Unvalidated stochastic dropout in behavior policy")
        if config["gradient_checkpointing"]: self.pipe.unet.enable_gradient_checkpointing()
        # Stable config equality excludes Diffusers bookkeeping-only default lists.
        scheduler={k:v for k,v in dict(self.pipe.scheduler.config).items() if not k.startswith("_")}
        self.contract={"model":editor.config.to_dict(),"scheduler":scheduler,"algorithm":"DDPO_SF_v1",
            "instruction":INSTRUCTION,"config":copy.deepcopy(config),"state_dtype":"float32",
            "normalization":"coordinate_mean_then_time_mean_fixed", "reference":"behavior_snapshot_no_KL"}
        self.version=0
        self.unet_calls=0

    @torch.no_grad()
    def condition(self,record,instruction=INSTRUCTION):
        source=self.editor.prepare_image(record["image"])
        tensor=self.pipe.image_processor.preprocess(source).to(self.editor.device,self.editor.dtype)
        image=self.pipe.vae.encode(tensor).latent_dist.mode()  # native unscaled source mode
        hidden=self.pipe._encode_prompt(instruction,self.editor.device,1,True)
        assert self.pipe.unet.config.in_channels==8 and image.shape[1]==4 and hidden.shape[0]==3
        return {"image":image.detach(),"hidden":hidden.detach(),"source":source,
            "sample_id":record["sample_id"],"image_id":str(record["image_id"]),"instruction":instruction}

    def predict(self,latent,t,condition):
        inputs=cfg_inputs(self.pipe.scheduler.scale_model_input(latent,t),condition["image"])
        assert inputs.shape[0]==3 and inputs.shape[1]==8
        # Respect the declared base precision. In particular FP32 validation must
        # never silently execute the UNet in BF16.
        inputs=inputs.to(self.editor.dtype)
        with torch.autocast(device_type=self.editor.device.type,dtype=torch.bfloat16,
                            enabled=self.editor.dtype==torch.bfloat16):
            if self.config.get("cfg_execution")=="sequential":
                prediction=torch.cat([self.pipe.unet(inputs[i:i+1],t,encoder_hidden_states=condition["hidden"][i:i+1],return_dict=False)[0] for i in range(3)])
                self.unet_calls+=3
            else:
                prediction=self.pipe.unet(inputs,t,encoder_hidden_states=condition["hidden"],return_dict=False)[0]
                self.unet_calls+=1
        # Guidance remains FP32 outside autocast in both sampling and training.
        return guided_prediction(prediction.float(),self.editor.config.guidance_scale,self.editor.config.image_guidance_scale)

    @torch.no_grad()
    def sample(self,condition,seed,trajectory_id):
        started=time.perf_counter()
        generator=torch.Generator(device=self.editor.device).manual_seed(seed)
        latent=torch.randn((1,4,*condition["image"].shape[-2:]),generator=generator,device=self.editor.device,dtype=torch.float32)
        initial=latent.clone(); steps=[]
        for t in self.pipe.scheduler.timesteps:
            noise=torch.randn(latent.shape,generator=generator,device=latent.device,dtype=torch.float32)
            action,logp,mean,std=transition(self.pipe.scheduler,self.predict(latent,t,condition),t,latent,
                self.config["eta"],noise=noise)
            steps.append({"t":int(t),"state":latent.cpu(),"action":action.cpu(),"noise":noise.cpu(),
                "logp":logp.cpu(),"std":float(std)})
            latent=action
        decoded=self.pipe.vae.decode((latent/self.pipe.vae.config.scaling_factor).to(self.editor.dtype),return_dict=False)[0]
        image=self.pipe.image_processor.postprocess(decoded,output_type="pil")[0]
        if self.editor.device.type == "cuda":
            torch.cuda.synchronize(self.editor.device)
        return image,{"trajectory_id":trajectory_id,"policy_version":self.version,"contract":copy.deepcopy(self.contract),
            "sample_id":condition["sample_id"],"image_id":condition["image_id"],"instruction":condition["instruction"],
            "seed":seed,"initial_state":initial.cpu(),"steps":steps,"generation_seconds":time.perf_counter()-started}
