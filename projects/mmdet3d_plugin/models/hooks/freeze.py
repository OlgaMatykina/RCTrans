from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class FreezeAllButNewDepthHook(Hook):
    def before_train(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        for name, param in model.named_parameters():
            if 'depth_model' in name or 'radar_depth_model' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
