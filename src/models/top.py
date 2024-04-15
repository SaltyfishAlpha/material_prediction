import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

import material_prediction.src.models.swin as swin
import material_prediction.materialistic.src.models.model_utils_pl as model_utils
import material_prediction.src.loss_functions as loss_functions

from torchinfo import summary


class ResidualConvUnit_custom(torch.nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()
        self.groups = 1

        self.conv1 = torch.nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            groups=self.groups,
        )

        self.conv2 = torch.nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=self.groups,
        )

        self.activation = activation

        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)

        out = self.activation(out)
        out = self.conv2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom(torch.nn.Module):
    """Feature fusion block."""

    def __init__(
            self,
            features,
            activation,
            align_corners=True,
            upsample=True
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.align_corners = align_corners
        out_features = features

        self.groups = 1

        self.out_conv = torch.nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation)
        self.upsample = upsample
        self.layer_norm_1 = torch.nn.LayerNorm(out_features)
        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)
        if self.upsample:
            output = torch.nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
            )

        output = self.out_conv(output)
        return output


class Materoal_Prediction_Module(pl.LightningModule):
    def __init__(self, mconf, margs, use_swin=False):
        super().__init__()
        # test
        self.use_swin = use_swin
        # config
        self.output_channel = 5
        # materialistic module
        self.mtModule = model_utils.create_model(mconf, margs)
        self.mtModule.load_checkpoint(margs.checkpoint_dir, map_location=torch.device('cuda:2'))
        self.mtModule.eval()
        # fix parameters
        for k, param in self.mtModule.named_parameters():
            # print('k:', k)
            param.requires_grad = False

        out_channels = 256

        # swin module
        # def __init__(self, dim, input_resolution, num_heads, win_size=10, shift_size=0,
        #              mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
        #              act_layer=nn.GELU, norm_layer=nn.LayerNorm, token_projection='linear', token_mlp='leff',
        #              se_layer=False):

        self.swin_1 = swin.SIMTransformerBlock(out_channels, 8, (out_channels, out_channels), win_size=8)
        # self.swin_2 = swin.SIMTransformerBlock(out_channels, 8, (out_channels // 2, out_channels // 2), win_size=8)
        # self.swin_3 = swin.SIMTransformerBlock(out_channels, 8, (out_channels // 4, out_channels // 4), win_size=8)
        # self.swin_4 = swin.SIMTransformerBlock(out_channels, 8, (out_channels // 4, out_channels // 4), win_size=8)
        #
        # # fusion layers
        # self.fusion_1 = FeatureFusionBlock_custom(out_channels,
        #                                           torch.nn.ReLU(),
        #                                           align_corners=True)
        #
        # self.fusion_2 = FeatureFusionBlock_custom(out_channels,
        #                                           torch.nn.ReLU(),
        #                                           align_corners=True)
        #
        # self.fusion_3 = FeatureFusionBlock_custom(out_channels,
        #                                           torch.nn.ReLU(),
        #                                           align_corners=True)
        #
        # self.fusion_4 = FeatureFusionBlock_custom(out_channels,
        #                                           torch.nn.ReLU(),
        #                                           align_corners=True, upsample=False)
        #
        # output
        self.out_conv = torch.nn.Sequential(torch.nn.Linear(256, 256),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(256, 128),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128, 128),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128, self.output_channel))

        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=5, padding=2, bias=True),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(256, 128, kernel_size=5, padding=2, bias=True),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(128, self.output_channel, kernel_size=3, padding=1, bias=True),
                                         torch.nn.Sigmoid())

        # self.out_conv = torch.nn.Sequential(torch.nn.Conv2d(256, 256, kernel_size=1, bias=True),
        #                                     torch.nn.ReLU(),
        #                                     torch.nn.Conv2d(256, 128, kernel_size=1, bias=True),
        #                                     torch.nn.ReLU(),
        #                                     torch.nn.Conv2d(128, 128, kernel_size=1, bias=True),
        #                                     torch.nn.ReLU(),
        #                                     torch.nn.Conv2d(128, self.output_channel, kernel_size=1, bias=True),
        #                                     torch.nn.Sigmoid(),
        #                                     )

    def forward(self, x, ref_pos):
        B, C, H, W = x.shape
        with torch.no_grad():
            scores, _, _, _, _, context_embeddings_1, context_embeddings_2, context_embeddings_3, context_embeddings_4, _, _, _, _ = self.mtModule.net(
                x, ref_pos)

        # context_embeddings_1.shape b, 256, 256, 256
        # context_embeddings_2.shape b, 256, 128, 128
        # context_embeddings_3.shape b, 256, 64, 64
        # context_embeddings_4.shape b, 256, 64, 64

        # conv test
        if not self.use_swin:
            conv1 = self.conv1(context_embeddings_1)
            predictions = torch.nn.functional.interpolate(conv1, scale_factor=2, mode="bilinear",
                                                          align_corners=False)
        else:
            # swin blocks
            swin_layer_1 = self.swin_1(context_embeddings_1.permute(0, 2, 3, 1).reshape(B, -1, 256),
                                       cluster_mask=scores)
            unflattened_1 = swin_layer_1.permute(0, 2, 1).reshape(B, -1, 256, 256)
            midtemp = self.out_conv(unflattened_1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            predictions = torch.nn.functional.interpolate(midtemp, scale_factor=2, mode="bilinear",
                                                          align_corners=False)

        # # swin blocks
        # swin_layer_1 = self.swin_1(context_embeddings_1.permute(0, 2, 3, 1).reshape(B, -1, 256), cluster_mask=scores)
        # swin_layer_2 = self.swin_2(context_embeddings_2.permute(0, 2, 3, 1).reshape(B, -1, 256), cluster_mask=scores)
        # swin_layer_3 = self.swin_3(context_embeddings_3.permute(0, 2, 3, 1).reshape(B, -1, 256), cluster_mask=scores)
        # swin_layer_4 = self.swin_4(context_embeddings_4.permute(0, 2, 3, 1).reshape(B, -1, 256), cluster_mask=scores)
        #
        # # Unflattening the spatial token maps at the different scales from different blocks
        # unflattened_1 = swin_layer_1.permute(0, 2, 1).reshape(B, -1, 256, 256)
        # unflattened_2 = swin_layer_2.permute(0, 2, 1).reshape(B, -1, 128, 128)
        # unflattened_3 = swin_layer_3.permute(0, 2, 1).reshape(B, -1, 64, 64)
        # unflattened_4 = swin_layer_4.permute(0, 2, 1).reshape(B, -1, 64, 64)
        #
        # # convs
        # # unflattened_1 = self.conv1(context_embeddings_1)
        # # unflattened_2 = self.conv2(context_embeddings_2)
        # # unflattened_3 = self.conv3(context_embeddings_3)
        # # unflattened_4 = self.conv4(context_embeddings_4)
        #
        # # merge
        # path4 = self.fusion_4(unflattened_4)
        # path3 = self.fusion_3(unflattened_3, path4)
        # path2 = self.fusion_2(unflattened_2, path3)
        # path1 = self.fusion_1(unflattened_1, path2)
        #
        # # result
        # predictions = self.out_conv(path1.permute(0, 2, 3, 1)).reshape(B, -1, H, W)
        # # return predictions
        #
        # result departure
        albedoPred, roughPred, metalPred = predictions.split([3, 1, 1], dim=1)

        return predictions, scores, albedoPred, roughPred, metalPred


######################################
#   Top Module
######################################
class TopModule(pl.LightningModule):
    def __init__(self, mconf, margs, cfg, use_swin=False):
        super().__init__()
        # config
        self.cfg = cfg
        # net
        self.net = Materoal_Prediction_Module(mconf, margs, use_swin=use_swin)
        # loss
        self.perceptual_loss = loss_functions.PerceptualLoss()
        # log
        self.logsets = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        albedo = batch['img']['albedo']
        # normal = batch['img']['normal']
        rough = batch['img']['roughness']
        metal = batch['img']['metallic']
        # depth = batch['img']['depth']

        segAlb = batch['img']['segAlb']
        segMat = batch['img']['segMat']

        # segGeo = batch['img']['segGeo']

        im = batch['img']['im']

        ref_pos = batch['ref_pos']

        predictions, scores, albedoPred, roughPred, metalPred = self.net(im, ref_pos)

        albedoPred = torch.clamp(albedoPred, 0, 1)

        # depthPred = loss_functions.LSregress(depthPred * segGeo, depth * segGeo, depthPred)
        # depthPred = torch.clamp(depthPred, min=0)

        pixAlbNum = torch.sum(segAlb).item()
        pixMatNum = torch.sum(segMat).item()

        # pixGeoNum = torch.sum(segGeo).item()

        L2Err = torch.sum(
            (torch.log1p(albedoPred) - torch.log1p(albedo)) * (torch.log1p(albedoPred) - torch.log1p(albedo)) * segAlb
        ) / pixAlbNum / 3.0
        percErr = self.perceptual_loss(albedoPred, albedo, segAlb, layers=self.cfg.train.perceptual.albedo) # todo: remove
        albedoErr = 0.5 * (L2Err + percErr * self.cfg.train.perceptual.weight)

        # percErr = self.perceptual_loss(normalPred, normal, segGeo, layers=self.cfg.train.perceptual.normal)
        # normalErr = loss_functions.normal_loss(normalPred, normal, segGeo)
        # normalErr = 0.5 * (normalErr + percErr * self.cfg.train.perceptual.weight)

        roughErr = torch.sum((roughPred - rough) * (roughPred - rough) * segMat) / pixMatNum
        metalErr = torch.sum((metalPred - metal) * (metalPred - metal) * segMat) / pixMatNum

        matErr = roughErr + metalErr

        # roughPred = matPred[:, 0:1, ...]
        # metalPred = matPred[:, 1:2, ...]

        percErrRough = self.perceptual_loss(roughPred.expand(-1, 3, -1, -1), rough.expand(-1, 3, -1, -1), segMat,
                                            layers=self.cfg.train.perceptual.material)
        percErrMetal = self.perceptual_loss(metalPred.expand(-1, 3, -1, -1), metal.expand(-1, 3, -1, -1), segMat,
                                            layers=self.cfg.train.perceptual.material)

        matErr = 0.5 * ((percErrRough + percErrMetal) * self.cfg.train.perceptual.weight * 0.5 + matErr)

        # depthErr = torch.sum((torch.log(depthPred + 1) - torch.log(depth + 1)) * (
        #         torch.log(depthPred + 1) - torch.log(depth + 1)) * segGeo) / pixGeoNum

        loss = albedoErr + matErr
        # loss = albedoErr + normalErr + matErr + depthErr

        with torch.no_grad():
            self.log_dict({
                'train/total_loss': loss,
                'train/albedo_loss': albedoErr,
                'train/material_loss': matErr,
            })
            self.logsets.append({
                'loss': loss,
                'albedo_loss': albedoErr,
                'material_loss': matErr,
            })
        # return {
        #     'loss': loss,
        #     'albedo_loss': albedoErr,
        #     'normal_loss': normalErr,
        #     'material_loss': matErr,
        #     'depth_loss': depthErr
        # }
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        albedo = batch['img']['albedo']  # 3
        rough = batch['img']['roughness']  # 1
        metal = batch['img']['metallic']  # 1

        # normal = batch['img']['normal']  # 3
        # depth = batch['img']['depth']  # 1

        segAlb = batch['img']['segAlb']  # 1
        segMat = batch['img']['segMat']  # 1

        # segGeo = batch['img']['segGeo']  # 1

        im = batch['img']['im']

        ref_pos = batch['ref_pos']

        predictions, scores, albedoPred, roughPred, metalPred = self.net(im, ref_pos)

        albedoPred = torch.clamp(albedoPred, 0, 1)

        # depthPred = loss_functions.LSregress(depthPred * segGeo, depth * segGeo, depthPred)
        # depthPred = torch.clamp(depthPred, min=0)

        pixAlbNum = torch.sum(segAlb).item()
        pixMatNum = torch.sum(segMat).item() # TODO: isZero?

        # pixGeoNum = torch.sum(segGeo).item()

        albedoErr = torch.sum((albedoPred - albedo) * (albedoPred - albedo) * segAlb) / pixAlbNum / 3.0

        # normalErr = torch.sum((normalPred - normal) * (normalPred - normal) * segGeo) / pixGeoNum / 3.0

        roughErr = torch.sum((roughPred - rough) * (roughPred - rough) * segMat) / pixMatNum
        metalErr = torch.sum((metalPred - metal) * (metalPred - metal) * segMat) / pixMatNum

        # depthErr = torch.sum((torch.log(depthPred + 1) - torch.log(depth + 1)) * (
        #         torch.log(depthPred + 1) - torch.log(depth + 1)) * segGeo) / pixGeoNum

        albLoss = albedoErr
        # normLoss = normalErr
        matLoss = (roughErr + metalErr)
        # depthLoss = depthErr

        totalErr = albLoss + matLoss

        self.log('loss', totalErr, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        return {
            'loss': totalErr,
            'albedo_loss': albedoErr,
            'material_loss': matLoss,
            'roughness_loss': roughErr,
            'metallic_loss': metalErr,
        }

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        albedo_loss = torch.stack([x['albedo_loss'] for x in outputs]).mean()
        material_loss = torch.stack([x['material_loss'] for x in outputs]).mean()
        roughness_loss = torch.stack([x['roughness_loss'] for x in outputs]).mean()
        metallic_loss = torch.stack([x['metallic_loss'] for x in outputs]).mean()
        self.log_dict({
            'validate/loss': loss,
            'validate/albedo_loss': albedo_loss,
            'validate/material_loss': material_loss,
            'validate/roughness_loss': roughness_loss,
            'validate/metallic_loss': metallic_loss,
        })

    def training_epoch_end(self, outputs):
        loss = torch.stack([x['loss'] for x in self.logsets]).mean()
        albedo_loss = torch.stack([x['albedo_loss'] for x in self.logsets]).mean()
        material_loss = torch.stack([x['material_loss'] for x in self.logsets]).mean()
        with torch.no_grad():
            self.log_dict({
                'train_total/loss': loss,
                'train_total/albedo_loss': albedo_loss,
                'train_total/material_loss': material_loss,
            })

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), **self.cfg.optim)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.5, -1, verbose=True)
        return [optimizer], [scheduler]
