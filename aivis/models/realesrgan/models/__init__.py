import importlib
from basicsr.utils import scandir
from os import path as osp

# automatically scan and import model modules for registry
# scan all the files that end with '_model.py' under the model folder
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(model_folder) if v.endswith('_model.py')]
# import all the model modules
# compute the current package name dynamically
package_name = __name__
# import all the model modules
_model_modules = [
    importlib.import_module(f'.{file_name}', package=package_name)
    for file_name in model_filenames
]
