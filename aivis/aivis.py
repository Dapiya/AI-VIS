"""AI-VIS module"""
import gc
import os
import numpy as np
import torch
import torchvision.transforms as transform
import skimage.transform as sktransform
from safetensors.torch import load_file

class AI_VIS:
    def __init__(self, gpu_id='0'):
        """Init AI-VIS module.

        Args:
            gpu_id (str): Set GPU device to use. If set -1, will disable GPU device enforcely.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1" if gpu_id is None else gpu_id
        self.gpu_id = gpu_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def load(self, weight_path='./aivis/weights', upscale=False, half_precision=False, tile=0, tile_pad=10, pre_pad=0, arch='1.5-large'):
        """Load AI-VIS model, including its module and weights.

        Args:
            weight_path (str): The path to the pretrained model.
            upscale (bool, optional): Whether to use super-resolution (Real-ESRGAN-x4-plus) model.
            half_precision (bool, optional): Whether to use half precision (fp16) on super-resolution model.
            tile (int, optional): Crop input images into tiles for saving GPU memory. See `models.realesrgan.RealESRGANer` for more information.
            tile_pad (int, optional): The pad size for each tile, to remove border artifacts.
            pre_pad (int, optional): Pad the input images to avoid border artifacts.
        """    
        self.upscale = upscale
        
        # load AI-VIS model weight and checkpoint
        if arch == '1.5-large':
            from .models.aivis_1_5_large import GeneratorUNet
            self._Gpath = os.path.join(weight_path, 'aivis_1_5_large.safetensors')
            self.G = GeneratorUNet().to(self.device)
            self._ckpt = load_file(self._Gpath)
            self.G.load_state_dict(self._ckpt)
            self.G.eval()
        elif arch == '1.5-small' or arch == '1.0':
            from .models.aivis_1_0 import GeneratorUNet # 1.0 and 1.5 small shares the same arch
            if arch == '1.5-small':
                self._Gpath = os.path.join(weight_path, 'aivis_1_5_small.safetensors')
            elif arch == '1.0':
                self._Gpath = os.path.join(weight_path, 'aivis_1_0.safetensors')
            self.G = GeneratorUNet().to(self.device)
            self._ckpt = load_file(self._Gpath)
            self.G.load_state_dict(self._ckpt)
            self.G.eval()
            
        self.T = transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.T_inverse = transform.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )
        if self.upscale:
            # load upscaling model (Real-ESRGAN-x4-plus)
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from .models.realesrgan import RealESRGANer
            self._Upath = os.path.join(weight_path, 'upscaler_1_5.safetensors')
            state_dict = load_file(os.path.join(weight_path, 'upscaler_1_5.safetensors'), device='cpu')
            self._Umodel = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            self._Umodel.load_state_dict(state_dict)
            self.U = RealESRGANer(
                model_path=self._Upath,
                model=self._Umodel,
                scale=4, dni_weight=None,
                tile=tile, tile_pad=tile_pad, pre_pad=pre_pad,
                half=half_precision, gpu_id=self.gpu_id
            )
            self.U.model = self._Umodel
            # use bilinear interpolation with skimage module
            # only use on lon/lat interpolation
            self.interpolate = lambda lonlat: (
                # x4 upscale from (500, 500) to (2000, 2000)
                sktransform.rescale(lonlat, 4, order=1)
            )
    
    def release(self):
        """Release GPU memory / RAM usage.
        """
        del self.G, self._Gpath, self._ckpt, self.T
        
        if self.upscale:
            del self.U, self._Upath, self._Umodel
        
        torch.cuda.empty_cache()
        gc.collect()
        
    def _build_input_tensor(self, datas, basemap,
                                sza, az, sat_za, sat_az):
            bt08, bt09, bt10, bt11, bt13, bt15, bt16 = datas

            bt13_08 = bt13 - bt08
            bt13_09 = bt13 - bt09
            bt13_10 = bt13 - bt10
            bt11_15 = bt11 - bt15
            bt13_15 = bt13 - bt15
            bt13_16 = bt13 - bt16

            bt13_10 = 1 - np.clip(bt13_10 + 12, 0, 74) / 74
            bt11_15 = 1 - np.clip(bt11_15 + 12, 0, 34) / 34
            bt13_16 = 1 - np.clip(bt13_16 + 3, 0, 44) / 44
            bt13_08 = 1 - np.clip(bt13_08 + 11, 0, 91) / 91
            bt13_09 = 1 - np.clip(bt13_09 + 10, 0, 80) / 80
            bt13_15 = 1 - np.clip(bt13_15 + 3, 0, 25) / 25
            bt13     = 1 - np.clip(bt13 + 103, 0, 148) / 148
            basemap  = basemap / 255.0

            # two white borders (reuse same shapes as original routine)
            side1 = np.ones([6, 500, 3])
            side2 = np.ones([512, 6, 3])

            sza_c    = np.full((500, 500), 1 - np.clip(sza,   0, 90) / 90)
            az_c     = np.full((500, 500), (np.clip(az,     -180, 180) + 180) / 360)
            sat_az_c = np.full((500, 500),  np.clip(sat_az,   0, 360) / 360)
            sat_za_c = np.full((500, 500),  np.clip(sat_za,   0, 90)  / 90)

            def _blk(rgb):
                x = np.concatenate([side1, rgb, side1], axis=0)
                x = np.concatenate([side2, x, side2], axis=1)
                return np.uint8(np.clip(x * 255, 0, 255))

            img = torch.cat([
                self.T(_blk(np.dstack([bt13,     bt13_15, az_c]))),
                self.T(_blk(np.dstack([bt13_10,  bt11_15, bt13_16]))),
                self.T(_blk(np.dstack([bt13_08,  bt13_09, sza_c]))),
                self.T(_blk(np.dstack([sat_az_c, sat_za_c, basemap])))
            ], dim=0)                                   # → [12,512,512]

            return img
        
    def _forward_batch(self, batch_tensors):
        imgs = torch.stack(batch_tensors).to(self.device)  # [B,12,512,512]
        with torch.no_grad():
            outs = self.T_inverse(self.G(imgs))
            outs = outs.permute(0, 2, 3, 1).cpu().numpy()  # [B,512,512,3]
            outs = outs[:, 6:-6, 6:-6, 0]                  # trim borders
        return outs
    
    def data_upscale(self, lons, lats, data):
        """Make a super-resolution to AI-VIS output data, and interpolate longitude and latitude to the same size.

        Args:
            lons (np.ndarray): Input longitude data to interpolate.
            lats (np.ndarray): Input latitude data to interpolate.
            data (np.ndarray): Input AI-VIS output data to Real-ESRGANx4 model.
        """
        if not lons.shape == lats.shape == data.shape == (500, 500):
            raise ValueError("Size must be (500, 500)")
        
        lons = self.interpolate(lons)
        lats = self.interpolate(lats)
        
        with torch.no_grad():
            data = np.stack((data.copy(), data.copy(), data.copy()), axis=2)
            data, _ = self.U.enhance(data * 255, outscale=4)
            data = data[:, :, 0] / 255
        
        return lons, lats, data
    
    def data_to_aivis(self, tile_iter, batch_size: int = 1, upscale: bool = False):
        """
        Convert one or many tiles to AI-VIS grayscale outputs.

        Parameters
        ----------
        tile_iter : Iterable[tuple]
            Each element must be a tuple of **exactly**:
              (lons, lats, datas, basemap, sza, az, sat_za, sat_az)
        batch_size : int, default 8
            Maximum number of tiles forwarded together in a single CUDA call.
        upscale : bool, default False
            If True, run Real-ESRGAN ×4 on each tile after inference.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            A list with the same ordering as `tile_iter`, where each element
            is ``(lons, lats, aivis_gray)``.
        """

        tiles = list(tile_iter)
        if not tiles:
            return []

        outputs: list[tuple] = []

        # Chunk → build tensors → forward → post-process
        for start in range(0, len(tiles), batch_size):
            chunk = tiles[start : start + batch_size]

            batch_tensors = []
            metas = []                 # (lon, lat) pairs kept in same order
            for (lon, lat, datas, bmap,
                 sza, az, sat_za, sat_az) in chunk:

                batch_tensors.append(
                    self._build_input_tensor(datas, bmap,
                                             sza, az, sat_za, sat_az)
                )
                metas.append((lon, lat))

            # UNet forward
            outs = self._forward_batch(batch_tensors)

            for (lon_, lat_), out in zip(metas, outs):
                outputs.append((lon_, lat_, out))

            # free GPU memory for long scenes
            del batch_tensors, outs
            torch.cuda.empty_cache()

        gc.collect()
        return outputs

