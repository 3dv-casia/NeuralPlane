import torch
import numpy as np
import trimesh

class TSDF():
    """ class to hold a truncated signed distance function (TSDF)

    Holds the TSDF volume along with meta data like voxel size and origin
    required to interpret the tsdf tensor.

    """

    def __init__(self, voxel_size, origin, tsdf_vol, attribute_vols=None,
                 attributes=None):
        """
        Args:
            voxel_size: metric size of voxels (ex: .04m)
            origin: origin of the voxel volume (xyz position of voxel (0,0,0))
            tsdf_vol: tensor of size hxwxd containing the TSDF values
            attribute_vols: dict of additional voxel volume data
                example: {'semseg':semseg} can be used to store a
                    semantic class id for each voxel
            attributes: dict of additional non voxel volume data (ex: instance
                labels, instance centers, ...)
        """

        self.voxel_size = voxel_size.reshape(1, 3)
        self.origin = origin.reshape(1, 3)
        self.tsdf_vol = tsdf_vol
        if attribute_vols is not None:
            self.attribute_vols = attribute_vols
        else:
            self.attribute_vols = {}
        if attributes is not None:
            self.attributes = attributes
        else:
            self.attributes = {}
        self.device = tsdf_vol.device

    def save(self, fname):
        data = {'origin': self.origin.cpu().numpy(),
                'voxel_size': self.voxel_size.cpu().numpy(),
                'tsdf': self.tsdf_vol.detach().cpu().numpy()}
        for key, value in self.attribute_vols.items():
            data[key] = value.detach().cpu().numpy()
        for key, value in self.attributes.items():
            data[key] = value.cpu().numpy()
        np.savez_compressed(fname, **data)

    @classmethod
    def load(cls, fname, voxel_types=None):
        """ Load a tsdf from disk (stored as npz).

        Args:
            fname: path to archive
            voxel_types: list of strings specifying which volumes to load
                ex ['tsdf', 'color']. tsdf is loaded regardless.
                to load all volumes in archive use None (default)

        Returns:
            TSDF
        """

        with np.load(fname) as data:
            voxel_size = torch.as_tensor(data['voxel_size']).view(1,3)
            origin = torch.as_tensor(data['origin']).view(1,3)
            tsdf_vol = torch.as_tensor(data['tsdf'])
            attribute_vols = {}
            attributes     = {}
            if 'color' in data and (voxel_types is None or 'color' in voxel_types):
                attribute_vols['color'] = torch.as_tensor(data['color'])
            if ('instance' in data and (voxel_types is None or
                                        'instance' in voxel_types or
                                        'semseg' in voxel_types)):
                attribute_vols['instance'] = torch.as_tensor(data['instance'])
            ret = cls(voxel_size, origin, tsdf_vol, attribute_vols, attributes)
        return ret

    def to(self, device):
        """ Move tensors to a device"""

        self.origin = self.origin.to(device)
        self.tsdf_vol = self.tsdf_vol.to(device)
        self.attribute_vols = {key:value.to(device)
                               for key, value in self.attribute_vols.items()}
        self.attributes = {key:value.to(device)
                           for key, value in self.attributes.items()}
        self.device = device
        return self