from .functions.deform_conv import deform_conv, modulated_deform_conv
from .functions.deform_pool import deform_roi_pooling
from .modules.deform_conv import (DeformConv, ModulatedDeformConv,
<<<<<<< HEAD
                                  ModulatedDeformConvPack)
=======
                                  DeformConvPack, ModulatedDeformConvPack)
>>>>>>> master-origin/master
from .modules.deform_pool import (DeformRoIPooling, DeformRoIPoolingPack,
                                  ModulatedDeformRoIPoolingPack)

__all__ = [
<<<<<<< HEAD
    'DeformConv', 'DeformRoIPooling', 'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'deform_conv',
    'modulated_deform_conv', 'deform_roi_pooling'
=======
    'DeformConv', 'DeformConvPack', 'ModulatedDeformConv',
    'ModulatedDeformConvPack', 'DeformRoIPooling', 'DeformRoIPoolingPack',
    'ModulatedDeformRoIPoolingPack', 'deform_conv', 'modulated_deform_conv',
    'deform_roi_pooling'
>>>>>>> master-origin/master
]
