from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class FreezeAllButNewDepthHook(Hook):
    def before_train(self, runner):
        model = model.module if hasattr(model, 'module') else runner.model

        for name, param in model.named_parameters():
            if 'img_backbone' in name or 'img_feats_compr' in name or 'img_neck' in name or 'dino_ms_fuse' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
