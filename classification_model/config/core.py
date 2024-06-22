from pathlib import Path
from typing import Sequence

from pydantic import BaseModel
from strictyaml import YAML,load

import classification_model

#project Directories
PACKAGE_ROOT=Path(classification_model.__file__).resolve().parent
ROOT=PACKAGE_ROOT.parent
CONFIG_FILE_PATH=PACKAGE_ROOT/"config.yml"
DATASET_DIR=PACKAGE_ROOT/"datasets"
TRAINED_MODEL_DIR=PACKAGE_ROOT/"trained_models"


class AppConfig(BaseModel):
    """
    Application-level config
    """
    package_name:str
    raw_data_file:str
    pipeline_save_file:str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    target:str
    unused_fields:Sequence[str]
    features:Sequence[str]
    test_size:float
    random_state:int
    max_depth:int
    n_estimators:int
    numerical_vars:Sequence[str]
    categorical_vars:Sequence[str]
    cabin_vars:Sequence[str]
    

class Config(BaseModel):
    """Master config"""

    app_config:AppConfig
    model_config:ModelConfig

def find_config_file()-> Path:
    """Locate the config file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")

def fetch_config_from_yaml(config_path:Path= None):
    """Pass yml containing the package configuration"""

    if not config_path:
        config_path=find_config_file()
    
    if config_path:
        with open(config_path,"r") as config_file:
            parsed_config=load(config_file.read())
            return parsed_config
    raise OSError(f"Did not find the config file at the path {config_path}")

def create_and_validate_config(parsed_config:YAML=None)->Config:
    """Run validation on config values"""
    if parsed_config is None:
        parsed_config=fetch_config_from_yaml()
    
    _config=Config(
        app_config=AppConfig(**parsed_config.data),
        model_config=ModelConfig(**parsed_config.data)
    )

    return _config

config=create_and_validate_config()