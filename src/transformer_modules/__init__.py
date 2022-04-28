import sys
import importlib
sys.modules['transformers.generation_utils']=importlib.import_module('src.transformer_modules.generation_utils')
